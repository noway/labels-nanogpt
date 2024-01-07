from toker import tokenize, tokens_to_array_of_numbers_without_full_vocab
from toker_decode import decode
import torch
from train import m
from toker_decode import decode_one_token
import yaml

with open('splits.json', 'r') as f:
    splits = eval(f.read())

with open('commonality_map.json', 'r') as f:
    commonality_map = eval(f.read())

with open('full_vocab.json', 'r') as f:
    full_vocab = eval(f.read())


with open('exercises0_0.yml', 'r') as f:
    data = yaml.safe_load(f)
    text = data['exercises']
    answer = data['answer']

print ('text',(text,))
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
the_answer = ''
with torch.no_grad():
    for token in m.module.generate(idx, 1000):
        token_str = decode_one_token(token)
        print(token_str, end='', flush=True)
        if token_str == '\n':
            break
        the_answer += token_str

print ('the_answer', (the_answer,))

is_correct = the_answer.strip() == str(answer)

print ('is_correct', (is_correct,))