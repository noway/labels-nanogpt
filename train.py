import string
import torch
import torch.nn as nn
from torch.nn import functional as F

with open('trainingdata.txt', 'r') as f:
    text = f.read()

chars = list(set(text))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

def encode(s):
    return [stoi[ch] for ch in s]
def decode(x):
    return ''.join([itos[i] for i in x])

batch_size = 32
block_size = 256
max_iters = 5000
num_embeddings = 32
device = 'mps'
learning_rate = 3e-4
eval_iters = 25
n_layer = 4
n_head = 4
dropout = 0.2


encoded = encode(text)
data = torch.tensor(encoded, dtype=torch.long)
data = data.to(device)

first_90_percent = int(len(data) * 0.9)
train_data = data[:first_90_percent]
val_data = data[first_90_percent:]

x = train_data[:block_size]
y = train_data[1:block_size+1]


def get_batch():
    data = train_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

def get_batch_val():
    data = val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

xb, yb = get_batch()

# for b in range(batch_size):
#     for t in range(block_size): 
#         context = xb[b,:t+1]
#         target = yb[b,t]
#         print(f"t={t} context={decode(context.tolist())} target={decode([target.item()])}")

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, num_embeddings)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """ a simple linear layer followed by non-linearity """

    def __init__(self, num_embeddings):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_embeddings, 4*num_embeddings),
            nn.ReLU(),
            nn.Linear(4*num_embeddings, num_embeddings),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, num_embeddings, n_head):
        super().__init__()
        head_size = num_embeddings // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(num_embeddings)
        self.ln1 = nn.LayerNorm(num_embeddings)
        self.ln2 = nn.LayerNorm(num_embeddings)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, num_embeddings)
        self.position_embedding_table = nn.Embedding(block_size, num_embeddings)
        self.blocks = nn.Sequential(*[Block(num_embeddings, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(num_embeddings) # final layer norm
        self.lm_head = nn.Linear(num_embeddings, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):

        B, T = idx.shape
        # idx and targets are both B x T
        tok_emb = self.token_embedding_table(idx) # B x T x C
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # T x C
        x = tok_emb + pos_emb # B x T x C
        x = self.blocks(x)
        x = self.ln_f(x)
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
        self.key = nn.Linear(num_embeddings, head_size, bias=False)
        self.query = nn.Linear(num_embeddings, head_size, bias=False)
        self.value = nn.Linear(num_embeddings, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

print (vocab_size)
m = BigramLanguageModel()
m.to(device)

print(sum(p.numel() for p in m.parameters() if p.requires_grad)/1e6, "M parameters")
optimizer = torch.optim.Adam(m.parameters(), lr=learning_rate)

for steps in range(max_iters):
    xb, yb = get_batch()
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if steps % eval_iters == 0:
        training_data_loss = loss.item()
        validation_batch = get_batch_val()
        logits, loss = m(*validation_batch)
        val_loss = loss.item()
        print(f"steps={steps} training_data_loss={training_data_loss} val_loss={val_loss}")

print(loss.item())

idx = torch.zeros(1, 1, dtype=torch.long)
idx = idx.to(device)
print(idx.shape)
print(decode(m.generate(idx, 1000)[0].tolist()))
