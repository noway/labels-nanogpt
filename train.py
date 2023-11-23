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

batch_size = 4
block_size = 8

x = train_data[:block_size]
y = train_data[1:block_size+1]

# print (x)
# print (y)

# for t in range(block_size): 
#     context = x[:t+1]
#     target = y[t]
#     print(f"t={t} context={decode(context.tolist())} target={decode([target.item()])}")

def get_batch():
    data = train_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    print (ix)

get_batch()