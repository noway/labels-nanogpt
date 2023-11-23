import string
import torch

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

print(train_data.shape, val_data.shape)