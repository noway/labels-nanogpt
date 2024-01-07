from toker import tokenize, tokens_to_array_of_numbers_without_full_vocab
from toker_decode import decode
import torch
from train import m
from toker_decode import decode_one_token
import yaml

with open('bigmodel/no_labels_toker/splits.json', 'r') as f:
    splits = eval(f.read())

with open('bigmodel/no_labels_toker/commonality_map.json', 'r') as f:
    commonality_map = eval(f.read())

with open('bigmodel/no_labels_toker/full_vocab.json', 'r') as f:
    full_vocab = eval(f.read())


def check_one_eval(eval_file):
    with open(eval_file, 'r') as f:
        data = yaml.safe_load(f)
        text = data['exercises']
        answer = data['answer']
        eval_type = data['eval_type']

    # print ('text',(text,))
    tokens = tokens_to_array_of_numbers_without_full_vocab(tokenize(text.lower(), splits, commonality_map), full_vocab)
    idx = torch.tensor(tokens).unsqueeze(0)

    # print (idx.shape)
    # print (decode(tokens), end='', flush=True)

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
            # print(token_str, end='', flush=True)
            if token_str == '\n' or token_str == ' ' or token_str == ',' or token_str == '.':
                break
            the_answer += token_str

    the_answer = the_answer.strip().replace('*', '')
    print ('the_answer', (the_answer,))
    is_correct = the_answer == str(answer)
    print ('is_correct', (is_correct,))
    return is_correct, eval_type


if __name__ == "__main__":
    correct_count = 0
    all_count = 0
    eval_type = ''
    for num1 in range(10):
        for num2 in range(10):
            if num1 < num2:
                continue
            file_path = f'exercises{num1}_{num2}.yml'
            print ('file_path', (file_path,))
            is_correct, eval_type = check_one_eval(file_path)
            if is_correct:
                correct_count += 1
            all_count += 1
    print ('eval_type', (eval_type,))
    print ('correct_count', (correct_count,))
    print ('all_count', (all_count,))