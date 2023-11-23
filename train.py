import string
import torch
import torch.nn as nn
from torch.nn import functional as F

stoi = { ch:i for i,ch in enumerate(string.printable) }
itos = { i:ch for i,ch in enumerate(string.printable) }
def encode(s):
    return [stoi[ch] for ch in s]
def decode(x):
    return ''.join([itos[i] for i in x])
with open('input.txt', 'r') as f:
    text = f.read()

encoded = encode(text)
data = torch.tensor(encoded, dtype=torch.long)

first_90_percent = int(len(data) * 0.9)
train_data = data[:first_90_percent]
val_data = data[first_90_percent:]

batch_size = 4
block_size = 8

x = train_data[:block_size]
y = train_data[1:block_size+1]


def get_batch():
    data = train_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

xb, yb = get_batch()

print(xb.shape, yb.shape)
print(xb[0])
print(yb[0])

for b in range(batch_size):
    for t in range(block_size): 
        context = xb[b,:t+1]
        target = yb[b,t]
        print(f"t={t} context={decode(context.tolist())} target={decode([target.item()])}")


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets):
        logits = self.token_embedding_table(idx)
        return logits
    

vocab_size = len(stoi)
print (vocab_size)
m = BigramLanguageModel(vocab_size)
out = m(xb, yb)
print (out.shape)