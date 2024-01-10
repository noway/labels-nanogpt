from train import m
import torch
from toker_decode import decode_one_token

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

with torch.no_grad():
    for token in m.module.generate(idx, 1000):
        token_str = decode_one_token(token)
        print(token_str, end='', flush=True)
