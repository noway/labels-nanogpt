from toker import tokenize, tokens_to_array_of_numbers_without_full_vocab
from toker_decode import decode
import torch
from train import m

text = """\_\_\_\_\_\_\_\_\_\_\_\_\_
2.  object: triangle, rectangle triangle, rectangle colored rectangle, rectangle. (what pattern would be kind to the pattern?)
3.  circle, triangle, circle, square, triangle, circle, \_\_\_\_\_\_, \_\_\_\_\_\_, \_\_\_\_\_\_\_, \_\_\_\_\_\_\_\_
4.  red, rectangle, triangle, circle, rectangle, square, triangle, rectangle, rectangle, \_\_\_\_\_\_\_, \_\_\_\_\_\_\_

**answers
to activity 3:**

1.  circle, square
2.  rectangle, rectangle
3.  hexagon, circle
4.  triangle, pentagon

## describing shape patterns

patterns can get trickier! some patterns increase or decrease in size or change colors and sequences. let's learn how to describe such patterns.

### activity 3: describe the pattern

tell us what the pattern is doing in each sequence below.

1.  small circle, big circle, small circle, big circle, big circle, \_\_\_\_\_\_
2.  blue square, blue square, red square, blue square, red square, \_\_\_
"""

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

