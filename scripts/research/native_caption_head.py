"""Train a residual scene head over learned proposals using only train captions."""

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path.cwd()))
from learned_center_fusion import FEATURES, MAX_REGIONS, N_ATTRS, Fusion
from native_centers import CenterDetector, GROUPS, HEIGHT, STRIDE, WIDTH, read_rgb, read_rows
from scripts import pixel_reasoning as pipeline


ART = Path("artefacts/research")
DATA = Path("data")
CACHE = ART / "native_caption"


@torch.inference_mode()
def make_cache(split, model, projection):
    _, scenes, _, _ = read_rows(split)
    names = list(scenes)
    features = np.zeros((len(names), MAX_REGIONS, FEATURES), np.float32)
    xy = np.zeros((len(names), MAX_REGIONS, 2), np.float32)
    valid = np.zeros((len(names), MAX_REGIONS), bool)
    dimensions = torch.tensor([WIDTH, HEIGHT])
    for start in range(0, len(names), 32):
        batch = names[start:start+32]
        pixels = np.stack([read_rgb(DATA/split/name) for name in batch])
        data = torch.from_numpy(pixels).permute(0,3,1,2).float()/255
        latent = model.features(data.contiguous(memory_format=torch.channels_last))
        raw = model.head(latent)
        p = raw[:,0].sigmoid()
        peaks = (p == F.max_pool2d(p[:,None],3,1,1)[:,0]) & (p>=.08)
        for j in range(len(batch)):
            ys,xs = torch.where(peaks[j])
            order = p[j,ys,xs].argsort(descending=True)[:20]
            ys,xs = ys[order],xs[order]
            if not len(order):
                continue
            mid = (raw[j,1:3,ys,xs].clip(-.5,1.5)+torch.stack((xs,ys))).T*STRIDE
            extent = raw[j,3:5,ys,xs].clip(-2,1.6).exp().T*24
            attr = [g.log_softmax(0).T for g in raw[j,5:,ys,xs].split([len(g) for g in GROUPS])]
            descriptor = latent[j,:,ys,xs].T @ projection
            features[start+j,:len(order)] = torch.cat((
                raw[j,0,ys,xs][:,None], *attr,
                (mid-.5*extent)/dimensions, extent/dimensions, descriptor,
            ),dim=1).numpy()
            xy[start+j,:len(order)] = ((mid+extent*torch.tensor([0.,.38]))/dimensions*2-1).numpy()
            valid[start+j,:len(order)] = True
    np.savez(CACHE/f"{split}.npz",names=names,x=features,xy=xy,valid=valid)
    print(split, "detections",int(valid.sum()),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ART / "native_natural_epoch21.pt")
    args = parser.parse_args()
    torch.set_num_threads(3)
    model = CenterDetector().to(memory_format=torch.channels_last).eval()
    model.load_state_dict(torch.load(args.checkpoint,weights_only=True,map_location="cpu"))
    teacher = torch.load(ART/"scene_silhouette.pt",weights_only=True,map_location="cpu")
    teacher["relation_offset"] = torch.zeros_like(teacher["relation_offset"])
    generator = torch.Generator().manual_seed(217)
    matrix = torch.randn(model.head.in_channels, 3+N_ATTRS, generator=generator)
    projection = torch.linalg.qr(matrix)[0]
    CACHE.mkdir(exist_ok=True)
    for split in ("train","val"):
        make_cache(split,model,projection)
    pipeline.ImageTable = lambda *_: Fusion(teacher)
    pipeline.run(DATA,epochs=6,batch_size=64,cache_dir=CACHE,posterior="joint",boundary=True,
                 device_name="cpu",report_path=ART/"region_native_caption_head.json")


if __name__ == "__main__":
    main()
