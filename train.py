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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, head_size, num_heads):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_embeddings, num_embeddings)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

class FeedForward(nn.Module):
    """ a simple linear layer followed by non-linearity """

    def __init__(self, num_embeddings):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_embeddings, 4*num_embeddings),
            nn.ReLU(),
            nn.Linear(4*num_embeddings, num_embeddings),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, num_embeddings, n_head):
        super().__init__()
        head_size = num_embeddings // n_head
        self.sa = MultiHeadAttention(head_size, n_head)
        self.ffwd = FeedForward(num_embeddings)

    def forward(self, x):
        x = x + self.sa(x)
        x = x + self.ffwd(x)
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, num_embeddings)
        self.position_embedding_table = nn.Embedding(block_size, num_embeddings)
        self.blocks = nn.Sequential(
            Block(num_embeddings, 4),
            Block(num_embeddings, 4),
            Block(num_embeddings, 4),
        )
        # self.sa_heads = MultiHeadAttention(4, num_embeddings // 4)
        # self.ffwd = FeedForward(num_embeddings)
        self.lm_head = nn.Linear(num_embeddings, vocab_size)

    def forward(self, idx, targets=None):

        B, T = idx.shape
        # idx and targets are both B x T
        tok_emb = self.token_embedding_table(idx) # B x T x C
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # T x C
        x = tok_emb + pos_emb # B x T x C
        # x = self.sa_heads(x) # B x T x C
        # x = self.ffwd(x) # B x T x C
        x = self.blocks(x)
        logits = self.lm_head(x) # B x T x vocab_size

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # this is an array for the generated tokens
        # we'll keep appending to it as we generate more tokens
        # we'll stop when we reach max_new_tokens
        idx_result = idx.clone().detach()
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx_result = torch.cat([idx_result, idx_next], dim=-1)
            idx = torch.cat([idx, idx_next], dim=-1)
            # clip idx to block_size so that we're not feeding the model tokens past it's context window
            idx = idx[:, -block_size:]
        return idx_result

    
class Head(nn.Module):
    """ the head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(num_embeddings, head_size)
        self.query = nn.Linear(num_embeddings, head_size)
        self.value = nn.Linear(num_embeddings, head_size)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) / C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        v = self.value(x)
        out = wei @ v
        return out

print (vocab_size)
m = BigramLanguageModel()
logits, loss = m(xb, yb)
# print (logits.shape)
# print (loss)


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

idx = torch.zeros(1, 1, dtype=torch.long)
print(idx.shape)
print(decode(m.generate(idx, 100)[0].tolist()))
