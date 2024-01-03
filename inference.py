from train import m
import torch

device = (
    'cuda'
    if torch.cuda.is_available()
    else 'mps'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    else 'cpu'
)

idx = torch.zeros(1, 1, dtype=torch.long)
idx = idx.to(device)
print(idx.shape)
print(m.module.generate(idx, 1000)[0].tolist())
