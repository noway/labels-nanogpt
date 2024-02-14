from train_simple import m
from toker_simple import tokenize, tokens_to_array_of_numbers_without_full_vocab
from toker_decode_simple import decode_one_token
import torch
import json
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
        tokenize(text.lower(), splits, commonality_map)[0] + (["@digit_tokens@"] if COMMONALITY_LABEL_ENABLED else []), full_vocab
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
    expected_token = tokens_to_array_of_numbers_without_full_vocab(tokenize(str(answer), splits, commonality_map)[0], full_vocab)[1 if COMMONALITY_LABEL_ENABLED else 0]
    # print ('expected_token', (expected_token,))
    the_probability = 0
    with torch.no_grad():
        the_probability = m.module.generate(idx, 1000, expected_token)
        # for token in m.module.generate(idx, 1000):
        #     token_str = decode_one_token(token)
        #     if (
        #         token_str == '\n'
        #         or token_str == ' '
        #         or token_str == ','
        #         or token_str == '.'
        #         or token_str == '\\'
        #     ):
        #         break
        #     the_answer += token_str

    the_answer = the_answer.strip().replace('*', '')
    the_answer = (re.findall(r'^\d+', the_answer) + [''])[0]
    if the_answer == 'five':
        the_answer = '5'
    # print('the_answer', (the_answer,))
    is_correct = the_answer == str(answer)
    # print('is_correct', (is_correct,))
    print('.', end='', flush=True)
    return the_probability, eval_type


if __name__ == '__main__':
    # correct_count = 0
    all_count = 0
    eval_type = ''
    probability_sum = 0
    probability_dict = {}
    for num1 in range(10):
        for num2 in range(10):
            if num1 < num2:
                continue
            file_path = f'exercises{num1}_{num2}.yml'
            # print('file_path', (file_path,))
            the_probability, eval_type = check_one_eval(file_path)
            probability_sum += the_probability
            probability_dict[file_path] = the_probability
            # if is_correct:
            #     correct_count += 1
            all_count += 1
    print()
    print('eval_type', (eval_type,))
    print('probability_sum', (probability_sum,))
    print('all_count', (all_count,))

    with open(f'eval_results{suffix}-{eval_type}.json', 'w') as f:
        json_of_all_above = {
            'eval_type': eval_type,
            'probability_sum': probability_sum,
            'all_count': all_count,
            'probability_dict': probability_dict,
        }
        json_str = json.dumps(json_of_all_above)
        f.write(json_str)