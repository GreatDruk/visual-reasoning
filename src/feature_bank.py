"""Cache one frozen spatial representation on the GPU, with no training-time JPEG/CNN work."""

import numpy as np
import torch

from .runtime import Progress


class FrozenFeatures:
    @torch.no_grad()
    def __init__(self, model, scenes, device):
        model.eval()
        batch = max(8, 64 // model.scale**2) if device.type == "cuda" else 2
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        self.features = None
        finite = torch.ones((), dtype=torch.bool, device=device)
        progress = Progress(f"{scenes.split}/frozen_features", len(scenes.names), device)
        for first in range(0, len(scenes.names), batch):
            indices = np.arange(first, min(first + batch, len(scenes.names)))
            value = model.extract(scenes.images(indices, device)).to(dtype)
            finite &= torch.isfinite(value).all()
            if self.features is None:
                shape = (len(scenes.names), *value.shape[1:])
                required = int(np.prod(shape)) * torch.empty((), dtype=dtype).element_size()
                if device.type == "cuda":
                    free, _ = torch.cuda.mem_get_info(device)
                    if required + 2 * 2**30 > free:
                        raise MemoryError(
                            "Frozen feature cache needs its allocation plus 2 GiB for readout training"
                        )
                self.features = torch.empty(
                    shape, dtype=dtype, device=device, memory_format=torch.channels_last
                )
            self.features[first : first + len(indices)].copy_(value)
            progress.update(first + len(indices), batch=batch)
        if not finite:
            raise FloatingPointError("Non-finite cached CNN features")
        self.bytes = self.features.numel() * self.features.element_size()
        progress.update(
            len(scenes.names),
            force=True,
            feature_shape=list(self.features.shape),
            cache_gib=self.bytes / 2**30,
        )
