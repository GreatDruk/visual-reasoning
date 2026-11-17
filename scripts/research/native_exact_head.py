"""Calibrate the native detector from official train count captions."""

from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path.cwd()))
from native_centers import CenterDetector, ROOT, count_loss, evaluate, read_rgb, read_rows, training_counts


ART = Path("artefacts/research")


def main():
    torch.set_num_threads(3)
    model = CenterDetector().to(memory_format=torch.channels_last).eval()
    model.load_state_dict(torch.load(ART/"native_natural_epoch21.pt",weights_only=True,map_location="cpu"))
    model.features.requires_grad_(False)
    rgb, scene, filters, labels = training_counts()
    print("Official exact train captions",len(labels),flush=True)
    latent = None
    with torch.no_grad():
        for start in range(0,len(rgb),16):
            x = torch.from_numpy(rgb[start:start+16]).permute(0,3,1,2).float()/255
            local = model.features(x.contiguous(memory_format=torch.channels_last))
            if latent is None:
                latent = torch.empty((len(rgb), *local.shape[1:]), dtype=torch.float16,
                                     memory_format=torch.channels_last)
            latent[start:start+len(local)].copy_(local)
    del rgb
    rng = np.random.default_rng(319)
    optimizer = torch.optim.AdamW(model.head.parameters(),lr=.00035)
    prior = {name:tensor.detach().clone() for name,tensor in model.head.state_dict().items()}
    teacher = torch.nn.Conv2d(model.head.in_channels,model.head.out_channels,1)
    teacher.load_state_dict(prior)
    teacher.requires_grad_(False)
    for epoch in range(1,5):
        history = []
        order = rng.permutation(len(labels))
        for selected in np.array_split(order,max(1,len(order)//16)):
            features = latent[scene[selected]].float().contiguous(memory_format=torch.channels_last)
            raw = model.head(features)
            prior_raw = teacher(features)
            loss = count_loss(raw,torch.as_tensor(filters[selected]),torch.as_tensor(labels[selected]))
            loss = loss + .12*F.mse_loss(raw,prior_raw)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            history.append(float(loss.detach()))
        print(epoch,"train loss",round(float(np.mean(history)),6),flush=True)
    del latent
    table = read_rows("val")
    val = (*table,np.stack([read_rgb(ROOT/"val"/name) for name in table[1]]))
    evaluate(model,torch.device("cpu"),val,"caption_exact_head")


if __name__=="__main__":
    main()
