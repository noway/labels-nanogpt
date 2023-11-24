import string
import torch
import torch.nn as nn
from torch.nn import functional as F

with open('input.txt', 'r') as f:
    text = f.read()

chars = list(set(text))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

def encode(s):
    return [stoi[ch] for ch in s]
def decode(x):
    return ''.join([itos[i] for i in x])

encoded = encode(text)
data = torch.tensor(encoded, dtype=torch.long)

first_90_percent = int(len(data) * 0.9)
train_data = data[:first_90_percent]
val_data = data[first_90_percent:]

batch_size = 32
block_size = 8
num_embeddings = 32

x = train_data[:block_size]
y = train_data[1:block_size+1]


def get_batch():
    data = train_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

xb, yb = get_batch()

# print(xb.shape, yb.shape)
# print(xb[0])
# print(yb[0])

for b in range(batch_size):
    for t in range(block_size): 
        context = xb[b,:t+1]
        target = yb[b,t]
        print(f"t={t} context={decode(context.tolist())} target={decode([target.item()])}")


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, num_embeddings)

    def forward(self, idx, targets=None):

        logits = self.token_embedding_table(idx)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=-1)
        return idx

    

print (vocab_size)
m = BigramLanguageModel()
logits, loss = m(xb, yb)
# print (logits.shape)
# print (loss)

idx = torch.zeros(1, 1, dtype=torch.long)

optimizer = torch.optim.Adam(m.parameters(), lr=1e-3)

for steps in range(10000):
    xb, yb = get_batch()
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if steps % 100 == 0:
        print(loss.item())

print(loss.item())

print(decode(m.generate(idx, 100)[0].tolist()))
