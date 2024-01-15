import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import datetime
import shutil
import signal
import time


with open('tokens.json', 'r') as f:
    json_str = f.read()
encoded = eval(json_str)

with open('labels.json', 'r') as f:
    json_str = f.read()
encoded_labels = eval(json_str)

chars = list(set(encoded))
vocab_size = len(chars)

labels_set_list = list(set(encoded_labels))
label_size = len(labels_set_list)
print('label_size', label_size)

batch_size = 18
block_size = 1024
checkpoint1_sec = 11700
total_train_sec = 23400
num_embeddings = 512
device = (
    'cuda'
    if torch.cuda.is_available()
    else 'mps'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    else 'cpu'
)
print('device', device)
learning_rate = 3e-4
eval_iters = 25
n_layer = 12
n_head = 16
dropout = 0.2

compute_unit_count = torch.cuda.device_count() if device == 'cuda' else 1

data = torch.tensor(encoded, dtype=torch.long)
data = data.to(device)
print('data', data.shape)

data_labels = torch.tensor(encoded_labels, dtype=torch.long)
data_labels = data_labels.to(device)
print('data_labels', data_labels.shape)

first_90_percent = int(len(data) * 0.9)
train_data = data[:first_90_percent]
val_data = data[first_90_percent:]

train_data_labels = data_labels[:first_90_percent]
val_data_labels = data_labels[first_90_percent:]


def compute_ix_one(i, block_size, data):
    return (i * block_size) % (len(data) - block_size)


def compute_ix(j, block_size, data):
    return torch.tensor(
        [
            compute_ix_one(i + batch_size * j, block_size, data)
            for i in range(batch_size)
        ]
    )


j = 0


def get_batch():
    global j
    data = train_data
    data_labels = train_data_labels
    ix = compute_ix(j, block_size, data)
    j += 1
    x = torch.stack([data[i : i + block_size] for i in ix])
    labels = torch.stack([data_labels[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, labels, y


def get_batch_val():
    data = val_data
    data_labels = val_data_labels
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    labels = torch.stack([data_labels[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, labels, y


class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

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
    """a simple linear layer followed by non-linearity"""

    def __init__(self, num_embeddings):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_embeddings, 4 * num_embeddings),
            nn.ReLU(),
            nn.Linear(4 * num_embeddings, num_embeddings),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation"""

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
        self.label_embedding_table = nn.Embedding(label_size, num_embeddings)
        self.blocks = nn.Sequential(
            *[Block(num_embeddings, n_head=n_head) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(num_embeddings)  # final layer norm
        self.lm_head = nn.Linear(num_embeddings, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, idx_labels, targets=None):
        B, T = idx.shape
        # C is the embedding dimension (size)
        # T is the number of tokens in the sequence (context window)
        # B is the batch size
        # idx and targets are both B x T
        tok_emb = self.token_embedding_table(idx)  # B x T x C
        label_emb = self.label_embedding_table(idx_labels)  # B x T x C
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # T x C
        x = tok_emb + label_emb + pos_emb  # B x T x C
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # B x T x vocab_size

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, idx_labels, max_new_tokens):
        # this is an array for the generated tokens
        # we'll keep appending to it as we generate more tokens
        # we'll stop when we reach max_new_tokens
        from toker_decode import decode_one_token, vectorize_label_with_map
        from toker import special_token_to_label_mapper

        self.eval()
        for _ in range(max_new_tokens):
            logits, _ = self(idx, idx_labels, targets=None)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            # idx_next = torch.multinomial(probs, num_samples=1)
            idx_next = torch.argmax(probs, dim=-1, keepdim=True)
            label_str = special_token_to_label_mapper(
                decode_one_token(idx_next[0][0].item())
            )
            if label_str is None:
                return False  # labeler for non-special tokens is not implemented
            idx_label_next = vectorize_label_with_map(label_str)
            values, indices = torch.topk(probs, 5)
            yield idx_next[0][0].item()
            idx = torch.cat([idx, idx_next], dim=-1)
            idx_labels = torch.cat(
                [
                    idx_labels,
                    torch.tensor([[idx_label_next]], dtype=torch.long, device=device),
                ],
                dim=-1,
            )
            # clip idx to block_size so that we're not feeding the model tokens past it's context window
            idx = idx[:, -block_size:]
            idx_labels = idx_labels[:, -block_size:]
        return True


class Head(nn.Module):
    """the head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(num_embeddings, head_size, bias=False)
        self.query = nn.Linear(num_embeddings, head_size, bias=False)
        self.value = nn.Linear(num_embeddings, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out


print(vocab_size)
m = nn.DataParallel(BigramLanguageModel())
m.to(device)

print(sum(p.numel() for p in m.parameters() if p.requires_grad) / 1e6, 'M parameters')


def get_timestring():
    return (
        datetime.datetime.now().strftime('%d%b%Y-%I%M%S%p-')
        + str(datetime.datetime.now().astimezone().tzinfo)
    ).lower()


def get_path(checkpoint=False):
    name = 'bigmodel/model_weights_with_label_embedding_consttime'
    if checkpoint:
        return f'{name}_{get_timestring()}.pth'
    return f'{name}.pth'


def timestamp():
    return datetime.datetime.now().timestamp()


PATH = get_path()

if os.path.dirname(PATH) != '':
    os.makedirs(os.path.dirname(PATH), exist_ok=True)

if os.path.exists(PATH):
    print('Loading model weights')
    m.module.load_state_dict(torch.load(PATH, map_location=torch.device(device)))
else:
    print('No model weights found')

if __name__ == '__main__':
    start_timestamp = timestamp()
    made_checkpoint = False
    optimizer = torch.optim.Adam(m.parameters(), lr=learning_rate)
    steps = 0

    def handler(signum, frame):
        print('Signal handler called with signal', signum)
        CHECKPOINT_PATH = get_path(checkpoint=True)
        print(f'Checkpoint model to {CHECKPOINT_PATH}')
        torch.save(m.module.state_dict(), CHECKPOINT_PATH)
        print(f'Copy model to {PATH}')
        shutil.copyfile(CHECKPOINT_PATH, PATH)
        print('Sleep 1 second')
        time.sleep(1)

    signal.signal(signal.SIGUSR1, handler)

    while True:
        xb, labelsb, yb = get_batch()
        logits, loss = m(xb, labelsb, yb)
        optimizer.zero_grad(set_to_none=True)
        (loss.sum() / compute_unit_count).backward()
        optimizer.step()
        if steps % eval_iters == 0:
            training_data_loss = (loss.sum() / compute_unit_count).item()
            validation_batch = get_batch_val()
            logits, loss = m(*validation_batch)
            val_loss = (loss.sum() / compute_unit_count).item()
            print(
                f'steps={steps} training_data_loss={training_data_loss} val_loss={val_loss}'
            )
            if timestamp() - start_timestamp > checkpoint1_sec and not made_checkpoint:
                CHECKPOINT_PATH = get_path(checkpoint=True)
                print(f'Checkpoint model to {CHECKPOINT_PATH}')
                torch.save(m.module.state_dict(), CHECKPOINT_PATH)
                print(f'Copy model to {PATH}')
                shutil.copyfile(CHECKPOINT_PATH, PATH)
                made_checkpoint = True
            if timestamp() - start_timestamp > total_train_sec:
                break
        steps += 1

    print(loss.item())

    CHECKPOINT_PATH = get_path(checkpoint=True)
    print(f'Checkpoint model to {CHECKPOINT_PATH}')
    torch.save(m.module.state_dict(), CHECKPOINT_PATH)
    print(f'Copy model to {PATH}')
    shutil.copyfile(CHECKPOINT_PATH, PATH)
