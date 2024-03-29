from toker_simple import tokenize, tokens_to_array_of_numbers_without_full_vocab
from toker_decode_simple import decode, decode_one_token
import torch
from train_simple import m
import sys

COMMONALITY_LABEL_ENABLED = len(sys.argv) > 1 and sys.argv[1] == 'with_labels'
suffix = '-with_labels' if COMMONALITY_LABEL_ENABLED else '-no_labels'


# text = """
# 1. 1 + 1 = 2
# 2. 1 + 2 = 3
# 3. 1 + 3 = 4
# 4. 1 + 4 = 5
# 5. 1 + 5 = 6
# 6. 1 + 6 = 7
# 7. 1 + 7 = 8
# 8. 1 + 8 = 9
# 9. 1 + 9 = 10
# 10. 1 + 10 = 11
# 11. 1 + 11 = 12
# 12. 1 + 13 = """

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


# text = """
# 1. **Exercise**: 5 + 1 = ?
# Let's count:
# 1, 2, 3, 4, 5
# Count 1 more:
# 6
# **Answer**: 5 + 1 = 6

# 2. **Exercise**: 5 + 2 = ?
# Let's count:
# 1, 2, 3, 4, 5
# Count 2 more:
# 6, 7
# **Answer**: 5 + 2 = 7

# 3. **Exercise**: 5 + 3 = ?
# Let's count:
# 1, 2, 3, 4, 5
# Count 3 more:
# 6, 7, 8
# **Answer**: 5 + 3 = 8

# 4. **Exercise**: 3 + 4 = ?
# Let's count:
# 1, 2, 3
# Count 4 more:
# 4, 5, 6, 7
# **Answer**: 3 + 4 = """


text = """
1. **Exercise**: 5 - 1 = ?
Let's count:
1, 2, 3, 4, 5
Count 1 less:
4
**Answer**: 5 - 1 = 4

2. **Exercise**: 5 - 2 = ?
Let's count:
1, 2, 3, 4, 5
Count 2 less:
4, 3
**Answer**: 5 - 2 = 3

3. **Exercise**: 5 - 3 = ?
Let's count:
1, 2, 3, 4, 5
Count 3 less:
4, 3, 2
**Answer**: 5 - 3 = 2

4. **Exercise**: 9 - 4 = ?
Let's count:
1, 2, 3, 4, 5, 6, 7, 8, 9
Count 4 less:
8, 7, 6, 5
**Answer**: 9 - 4 = """


with open(f'splits{suffix}.json', 'r') as f:
    splits = eval(f.read())

with open(f'commonality_map{suffix}.json', 'r') as f:
    commonality_map = eval(f.read())

with open(f'full_vocab{suffix}.json', 'r') as f:
    full_vocab = eval(f.read())

tokens = tokens_to_array_of_numbers_without_full_vocab(
    tokenize(text.lower(), splits, commonality_map)[0], full_vocab
)
idx = torch.tensor(tokens).unsqueeze(0)

print(idx.shape)

print(decode(tokens), end='', flush=True)


device = (
    'cuda'
    if torch.cuda.is_available()
    else 'mps'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    else 'cpu'
)

idx = idx.to(device)
with torch.no_grad():
    for token in m.module.generate(idx, 1000):
        token_str = decode_one_token(token)
        print(token_str, end='', flush=True)
        if token_str == '\n':
            break
