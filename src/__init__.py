import os

# Must be set before torch initialises MPS: ops without an MPS kernel fall back to CPU.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
