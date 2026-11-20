"""Fit ground-plane 3-D primitives to the observed image with a z-buffer."""

import math

import torch
from torch.nn import functional as F


def camera(device, height, width):
    y, x = torch.meshgrid(
        (torch.arange(height, device=device) + .5) * (160 / height),
        (torch.arange(width, device=device) + .5) * (240 / width),
        indexing="ij",
    )
    pitch = math.radians(24)
    sin, cos = math.sin(pitch), math.cos(pitch)
    u, v = (x.flatten() - 120) / 225, (y.flatten() - 80) / 225
    return u, -sin - v * cos, cos - v * sin, sin, cos


def initialize(box, sin, cos):
    """Back-project the estimated foot of each box to y=0."""
    foot = box[..., :2] + box[..., 2:] * box.new_tensor([.5, .88])
    v = (foot[..., 1] - 80) / 225
    t = 3.3 / (sin + v * cos).clamp_min(.12)
    x = t * (foot[..., 0] - 120) / 225
    z = t * (cos - v * sin)
    radius = (t * box[..., 2] / 450).clamp(.045, 1.2)
    return torch.stack((x, z, radius.log(), torch.zeros_like(x)), dim=-1)


def project_scene(state):
    # The symbolic executor uses increasing coordinates for right and front.
    return torch.stack((state[..., 0] / 3.1, (7.5 - state[..., 1]) / 3.1), dim=-1)


def render(state, shape, color, active, rays):
    """Intersect perspective rays with finite spheres, cuboids, and cylinders."""
    rx, ry, rz, _, _ = rays
    x, z, log_radius, yaw = (part[..., None] for part in state.unbind(-1))
    r = log_radius.exp()
    ox, oy, oz = -x, 3.3 - r, -z
    angle_cos, angle_sin = yaw.cos(), yaw.sin()
    dx, dz = rx * angle_cos + rz * angle_sin, rz * angle_cos - rx * angle_sin
    bx, bz = ox * angle_cos + oz * angle_sin, oz * angle_cos - ox * angle_sin

    def slab(origin, direction):
        safe = torch.where(direction.abs() < 1e-6,
                           torch.where(direction < 0, -1e-6, 1e-6), direction)
        a, b = (-r - origin) / safe, (r - origin) / safe
        return torch.minimum(a, b), torch.maximum(a, b)

    def quadratic(a, b, c):
        disc = b.square() - a * c
        root = disc.clamp_min(0).sqrt()
        return (-b - root) / a, (-b + root) / a, disc >= 0

    y0, y1 = slab(oy, ry)
    x0, x1 = slab(bx, dx)
    z0, z1 = slab(bz, dz)
    near_box = torch.maximum(torch.maximum(x0, y0), z0)
    far_box = torch.minimum(torch.minimum(x1, y1), z1)
    a = rx.square() + rz.square()
    b = ox * rx + oz * rz
    c = ox.square() + oz.square() - r.square()
    near_sphere, far_sphere, sphere_hit = quadratic(a + ry.square(), b + oy * ry, c + oy.square())
    near_rad, far_rad, cylinder_hit = quadratic(a, b, c)
    near_cylinder = torch.maximum(near_rad, y0)
    far_cylinder = torch.minimum(far_rad, y1)
    cube, sphere = shape[..., None] == 0, shape[..., None] == 1
    near = torch.where(cube, near_box, torch.where(sphere, near_sphere, near_cylinder))
    far = torch.where(cube, far_box, torch.where(sphere, far_sphere, far_cylinder))
    hit = torch.where(cube, True, torch.where(sphere, sphere_hit, cylinder_hit))
    ground_depth = -3.3 / ry
    visible = active[..., None] & hit & (near > 0) & (near <= far) & (near <= ground_depth + .02)
    front, owner = torch.where(visible, near, torch.inf).min(dim=1)
    labels = color.gather(1, owner)
    return torch.where(front.isfinite(), labels, -1)


@torch.no_grad()
def reconstruct(box, probability, shape, color, pixel_evidence, seed, *, steps=(36, 20)):
    """Minimize visible foreground/color disagreement with a coordinate prior.

    The detector supplies fixed object identities and classes. Every trial re-renders
    the same primitives with different positions, scales and cube yaw. A trial is
    accepted independently for each image. No caption or split-level fitting enters
    reconstruction.
    """
    device = box.device
    generator = torch.Generator(device=device).manual_seed(seed)
    active = probability >= .22
    ray = camera(device, 40, 60)
    initial = initialize(box, *ray[-2:])
    state = initial.clone()
    before = after = None
    for level, (height, width, iterations) in enumerate(zip((40, 64), (60, 96), steps)):
        rays = camera(device, height, width)
        evidence = F.interpolate(pixel_evidence.float(), size=(height, width), mode="area").flatten(2)
        target = evidence.amax(1)
        weight = (target > .2).sum(1).clamp_min(30)

        def loss(candidate):
            image = render(candidate, shape, color, active, rays)
            matching = evidence.gather(1, image.clamp_min(0)[:, None]).squeeze(1)
            error = torch.where(image >= 0, .72 + .40 * target - 1.12 * matching, target)
            delta = (candidate - initial) / candidate.new_tensor([.55, .8, .35, 1.6])
            prior = (delta.square().sum(-1) * probability * active).sum(1)
            return error.sum(1) / weight + .025 * prior

        best = loss(state)
        initial_loss = loss(initial)
        if level:
            use_initial = initial_loss < best
            state = torch.where(use_initial[:, None, None], initial, state)
            best = torch.minimum(best, initial_loss)
        before = best if before is None else before
        for step in range(iterations):
            shrink = 1 - .7 * step / max(1, iterations - 1)
            scale = state.new_tensor([.24, .34, .17, .55]) * shrink * (.55 if level else 1)
            delta = torch.randn(state.shape, generator=generator, device=device) * scale
            if step % 3:
                indices = torch.randint(box.shape[1], (len(box), 1), generator=generator, device=device)
                delta = delta * (torch.arange(box.shape[1], device=device) == indices)[..., None]
            trial = state + delta * active[..., None]
            trial[..., 1].clamp_(2.6, 16)
            trial[..., 2].clamp_(-3.2, .25)
            score = loss(trial)
            improved = score < best
            state = torch.where(improved[:, None, None], trial, state)
            best = torch.minimum(best, score)
        if level:
            before, after = initial_loss, best
    # The control uses exactly the same primitive identities, projection, and frame.
    return project_scene(initial), project_scene(state), before, after
