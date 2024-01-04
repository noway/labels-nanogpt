from toker import tokenize, tokens_to_array_of_numbers_without_full_vocab
from toker_decode import decode
import torch
from train import m

text = """1. What is the sum of 5 + 3?
Answer: 8

2. What is the difference between 11 and 8?
Answer: 3

3. What is the product of 5 and 4?
Answer: 20

4. What is the quotient of 18 divided by 3?
Answer: 6

5. What is the sum of 7 + 9 + 4?
Answer: 20

6. What is the difference between 15 and 11?
Answer: 4

7. What is the product of 5 and 7?
Answer: 35

8. What is the quotient of 21 divided by 3?
Answer: 7

9. What is the sum of 6 + 8 + 3?
Answer: 17

10. What is the difference between 16 and 10?
Answer: 6

11. What is the product of 4 and 6?
Answer: 24

12. What is the quotient of 24 divided by 4?
Answer: 6

13. What is the sum of 10 + 4?
Answer: """

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

