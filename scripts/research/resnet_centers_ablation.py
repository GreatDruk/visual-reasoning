"""Compare random and ImageNet initialization with the same stride-four detector."""

import argparse
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet50

from native_centers import ROOT, batches, evaluate, load_train, loss_for, read_rows, read_rgb


class ResNetCenters(nn.Module):
    def __init__(self, weights=None, *, full=False):
        super().__init__()
        backbone = resnet50(weights=None)
        if weights is not None:
            backbone.load_state_dict(torch.load(weights, weights_only=True, map_location="cpu"))
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.full = full
        if full:
            self.layer3 = backbone.layer3
            self.layer4 = backbone.layer4
            self.lateral3 = nn.Conv2d(1024, 64, 1)
            self.lateral4 = nn.Conv2d(2048, 64, 1)
        self.lateral1 = nn.Conv2d(256, 64, 1)
        self.lateral2 = nn.Conv2d(512, 64, 1)
        self.head = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64),
                                  nn.SiLU(), nn.Conv2d(64, 20, 1))
        nn.init.constant_(self.head[-1].bias[0], -2.19)
        self.register_buffer("mean", torch.tensor([.485, .456, .406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([.229, .224, .225])[None, :, None, None])

    def forward(self, pixels):
        high = self.layer1(self.stem((pixels-self.mean)/self.std))
        low = self.layer2(high)
        levels = [self.lateral1(high), self.lateral2(low)]
        if self.full:
            lower = self.layer3(low)
            levels.extend((self.lateral3(lower), self.lateral4(self.layer4(lower))))
        combined = levels[-1]
        for fine in reversed(levels[:-1]):
            combined = fine + F.interpolate(combined, size=fine.shape[-2:], mode="bilinear", align_corners=False)
        return self.head(combined)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--scenes", type=int, default=768)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=812)
    parser.add_argument("--runtime-only", action="store_true")
    args = parser.parse_args()
    if args.scenes < 1 or args.epochs < 1 or args.batch_size < 1:
        parser.error("scenes, epochs and batch size must be positive")
    torch.set_num_threads(4)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    examples, _ = load_train()
    data = read_rows("val")
    val = (*data, np.stack([read_rgb(ROOT / "val" / name) for name in data[1]]))
    rng = np.random.default_rng(args.seed)
    orders = [rng.permutation(len(examples))[:args.scenes] for _ in range(args.epochs)]
    for variant, weights in (("random", None), ("imagenet", args.weights)):
        torch.manual_seed(args.seed)
        rng = np.random.default_rng(args.seed)
        model = ResNetCenters(weights).to(device=device, memory_format=torch.channels_last)
        opt = torch.optim.AdamW(model.parameters(), lr=.001)
        started = time.perf_counter()
        steps = 0
        for epoch, indices in enumerate(orders, 1):
            model.train()
            losses = []
            for start in range(0, len(indices), args.batch_size):
                part = indices[start:start+args.batch_size]
                images, peak, boxes, labels = batches(examples, (), part, rng, device, augment=False)
                predicted = model(images)
                assert predicted.shape[-2:] == (40, 60)
                loss = loss_for(predicted, peak, boxes, labels)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                losses.append(float(loss.detach()))
                steps += 1
                if steps == 4:
                    print(variant, device, "seconds per update", (time.perf_counter()-started)/4, flush=True)
                    if args.runtime_only:
                        break
            if args.runtime_only:
                break
            print(variant, epoch, "loss", np.mean(losses), "minutes", (time.perf_counter()-started)/60, flush=True)
            evaluate(model, device, val, f"resnet50_{variant}_epoch{epoch}")


if __name__ == "__main__":
    main()
