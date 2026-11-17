"""Save the selected model and frozen validation object distributions for ablations."""

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path.cwd()))
from scripts import pixel_reasoning as pipeline
from src.models.slot_program import EXCLUSIVE_GROUPS, WORD_INDEX


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", choices=("crop", "silhouette", "seams", "counts"))
    args = parser.parse_args()
    prefix = Path("artefacts/research") / f"scene_{args.variant}"
    original = pipeline.predict_joint

    @torch.no_grad()
    def capture(model, data, batch_size, seed):
        model.eval()
        tensors = {"ids": data.ids, "features": data.table[0].numpy(),
                   "offsets": model.relation_offset.cpu().numpy()}
        parts = []
        for _, region, _, _, _, _, _ in data.batches(batch_size):
            encoded = model.encode(*region)
            present, attributes = model.slot_logits(encoded)
            attrs = [
                attributes[..., [WORD_INDEX[word] for word in group]].softmax(-1)
                for group in EXCLUSIVE_GROUPS
            ]
            parts.append([present.sigmoid(), encoded["position"], encoded["valid"], *attrs])
        for key, arrays in zip(("p", "xy", "valid", "size", "color", "material", "shape"), zip(*parts)):
            tensors[key] = torch.cat(arrays).cpu().numpy()
        np.savez_compressed(prefix.with_suffix(".npz"), **tensors)
        torch.save(model.state_dict(), prefix.with_suffix(".pt"))
        return original(model, data, batch_size, seed)

    pipeline.predict_joint = capture
    crop = args.variant != "silhouette"
    seams = args.variant == "seams"
    pipeline.run(Path("data"), epochs=7 if crop else 19, schedule_epochs=20, batch_size=64,
                 boundary=True, anchors=True, primitives=not crop, crop_shapes=crop,
                 fragments=seams, seams=seams, count_supervision=args.variant == "counts",
                 posterior="joint", device_name="cpu", report_path=prefix.with_name(prefix.name + "_report.json"))


if __name__ == "__main__":
    main()
