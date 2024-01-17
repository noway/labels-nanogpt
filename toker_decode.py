import json

suffix = '-label_embeddings'

with open(f'full_vocab{suffix}.json', 'r') as f:
    full_vocab = json.load(f)


with open(f'labels_map{suffix}.json', 'r') as f:
    labels_map = json.load(f)


def decode_one_token(token):
    token = full_vocab[token]
    token = token[2:] if token.startswith('##') else token
    token = '' if token == '@extremely_common@' else token
    token = '' if token == '@very_common@' else token
    token = '' if token == '@moderately_common@' else token
    token = '' if token == '@less_common@' else token
    token = '' if token == '@rare@' else token
    token = '' if token == '@emoji_and_symbols_tokens@' else token
    token = '' if token == '@super_special_tokens@' else token
    token = '' if token == '@typographic_tokens@' else token
    token = '' if token == '@digit_tokens@' else token
    token = '' if token == '@split_explainer@' else token
    token = '' if token == '@word_filler@' else token
    return token


def vectorize_label_with_map(label):
    if label in labels_map:
        return labels_map[label]
    else:
        raise Exception(f'Label {label} is not in labels_map')


def vectorize_labels_with_map(labels):
    return [vectorize_label_with_map(label) for label in labels]


def decode(tokens):
    result = []
    for token in tokens:
        token = decode_one_token(token)
        if token:
            result.append(token)
    return ''.join(result)
