from train_simple import m
from toker_simple import tokenize, tokens_to_array_of_numbers_without_full_vocab
from toker_decode_simple import decode_one_token
import torch
import yaml
import re
import sys

COMMONALITY_LABEL_ENABLED = len(sys.argv) > 1 and sys.argv[1] == 'with_labels'
suffix = '-with_labels' if COMMONALITY_LABEL_ENABLED else '-no_labels'

with open(f'splits{suffix}.json', 'r') as f:
    splits = eval(f.read())

with open(f'commonality_map{suffix}.json', 'r') as f:
    commonality_map = eval(f.read())

with open(f'full_vocab{suffix}.json', 'r') as f:
    full_vocab = eval(f.read())


def check_one_eval(eval_file):
    with open(eval_file, 'r') as f:
        data = yaml.safe_load(f)
        text = data['exercises']
        answer = data['answer']
        eval_type = data['eval_type']

    tokens = tokens_to_array_of_numbers_without_full_vocab(
        tokenize(text.lower(), splits, commonality_map)[0], full_vocab
    )
    idx = torch.tensor(tokens).unsqueeze(0)

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
            if (
                token_str == '\n'
                or token_str == ' '
                or token_str == ','
                or token_str == '.'
            ):
                break
            the_answer += token_str

    the_answer = the_answer.strip().replace('*', '')
    the_answer = (re.findall(r'^\d+', the_answer) + [''])[0]
    # print('the_answer', (the_answer,))
    is_correct = the_answer == str(answer)
    # print('is_correct', (is_correct,))
    print('.', end='', flush=True)
    return is_correct, eval_type


if __name__ == '__main__':
    correct_count = 0
    all_count = 0
    eval_type = ''
    for num1 in range(10):
        for num2 in range(10):
            if num1 < num2:
                continue
            file_path = f'exercises{num1}_{num2}.yml'
            # print('file_path', (file_path,))
            is_correct, eval_type = check_one_eval(file_path)
            if is_correct:
                correct_count += 1
            all_count += 1
    print()
    print('eval_type', (eval_type,))
    print('correct_count', (correct_count,))
    print('all_count', (all_count,))
