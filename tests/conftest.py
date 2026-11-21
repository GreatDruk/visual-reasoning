import torch

# Small probability programs are faster and more reproducible without thread oversubscription.
torch.set_num_threads(2)
