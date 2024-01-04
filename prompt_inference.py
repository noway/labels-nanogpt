from toker import tokenize, tokens_to_array_of_numbers_without_full_vocab
from toker_decode import decode
import torch
from train import m

text = """
1. 1 + 1 = 2
2. 1 + 2 = 3
3. 1 + 3 = 4
4. 1 + 4 = 5
5. 1 + 5 = 6
6. 1 + 6 = 7
7. 1 + 7 = 8
8. 1 + 8 = 9
9. 1 + 9 = 10
10. 1 + 10 = 11
11. 1 + 11 = 12
12. 1 + 13 = """

# text = """1. 2 + 1 = 3
# 2. 2 + 2 = 4
# 3. 2 + 3 = 5
# 4. 2 + 4 = 6
# 5. 2 + 5 = 7
# 6. 2 + 6 = 8
# 7. 2 + 7 = 9
# 8. 2 + 8 = 10
# 9. 2 + 9 = 11
# 10. 2 + 10 = 12
# 11. 2 + 11 = 13
# 12. 3 + 3 = """

# text = """1. 2 + 1 = 3
# 2. 2 + 2 = 4
# 3. 2 + 3 = 5
# 4. 2 + 4 = 6
# 5. 2 + 5 = 7
# 6. 2 + 6 = 8
# 7. 2 + 7 = 9
# 8. 2 + 8 = 10
# 9. 2 + 9 = 11
# 10. 2 + 10 = 12
# 11. 2 + 11 = 13
# 12. 3 + 2 = """



with open('splits.json', 'r') as f:
    splits = eval(f.read())

with open('commonality_map.json', 'r') as f:
    commonality_map = eval(f.read())

with open('full_vocab.json', 'r') as f:
    full_vocab = eval(f.read())

tokens = tokens_to_array_of_numbers_without_full_vocab(tokenize(text.lower(), splits, commonality_map), full_vocab)
idx = torch.tensor(tokens).unsqueeze(0)

print (idx.shape)

print (decode(tokens), end='', flush=True)



device = (
    'cuda'
    if torch.cuda.is_available()
    else 'mps'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    else 'cpu'
)

idx = idx.to(device)
with torch.no_grad():
    m.module.generate(idx, 1000)

