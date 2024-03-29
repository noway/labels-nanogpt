import re
import json
from collections import Counter
from collections import defaultdict

emoji_and_symbols_tokens = [
    '1️⃣',
    '2️⃣',
    '3️⃣',
    '4️⃣',
    '5️⃣',
    '6️⃣',
    '7️⃣',
    '8️⃣',
    '9️⃣',
    '🔟',
    '🔢',
    '⚪',
    '⚫',
    '⚽',
    '⚾',
    '✂',
    '✅',
    '✈',
    '✋',
    '✏',
    '✦',
    '✨',
    '✪',
    '❄',
    '❌',
    '❏',
    '❤',
    '➡',
    '⬅',
    '⬜',
    '⬡',
    '⭐',
    '🌟',
    '🌱',
    '🌲',
    '🌳',
    '🌴',
    '🌵',
    '🌷',
    '🌸',
    '🌹',
    '🌺',
    '🌻',
    '🌼',
    '🌾',
    '🍀',
    '🍂',
    '🍃',
    '🍇',
    '🍉',
    '🍊',
    '🍌',
    '🍎',
    '🍏',
    '🍐',
    '🍒',
    '🍓',
    '🍕',
    '🍞',
    '🍦',
    '🍩',
    '🍪',
    '🍬',
    '🍳',
    '🍴',
    '🍽',
    '🎈',
    '🎉',
    '🎒',
    '🎨',
    '🎵',
    '🎸',
    '🏀',
    '🏎',
    '🏘',
    '🏠',
    '🏡',
    '🏢',
    '🏰',
    '🐁',
    '🐈',
    '🐌',
    '🐘',
    '🐙',
    '🐝',
    '🐟',
    '🐠',
    '🐤',
    '🐦',
    '🐭',
    '🐰',
    '🐱',
    '🐳',
    '🐵',
    '🐶',
    '🐷',
    '🐸',
    '🐻',
    '👉',
    '👟',
    '👧',
    '👨',
    '👩',
    '💦',
    '💧',
    '💼',
    '📏',
    '📖',
    '📘',
    '📚',
    '🔍',
    '🔎',
    '🔥',
    '🔮',
    '🔴',
    '🔵',
    '🔶',
    '🔺',
    '🔼',
    '🕊',
    '🕓',
    '🖍',
    '🖐',
    '😄',
    '😊',
    '🚀',
    '🚌',
    '🚒',
    '🚓',
    '🚕',
    '🚗',
    '🚙',
    '🚛',
    '🚜',
    '🚧',
    '🚲',
    '🛑',
    '🛴',
    '🟠',
    '🟡',
    '🟢',
    '🟣',
    '🟥',
    '🟦',
    '🟨',
    '🟩',
    '🤚',
    '🥕',
    '🥚',
    '🥣',
    '🥤',
    '🦆',
    '🦩',
    '🧀',
    '🧐',
    '🧒',
    '🧦',
    '🧸',
    '🪁',
    '●',
    '☀',
    '★',
    '☑',
    '✓',
    '↑',
    '→',
    '↓',
    '□',
    '▢',
    '△',
    'π',
]

super_special_tokens = [
    '------------------------------------------------------------------------',
    '\u200d',
    '\xa0',
    '\\<\\|image\\|\\>',
    '\\<\\|document\\|\\>',
    '\\<\\|unsolvedproblem\\|\\>',
    ' ',
    '\n',
    '\\',
    '️',
]

typographic_tokens = [
    '$',
    '*',
    '.',
    "'",
    '-',
    '#',
    '=',
    ':',
    '+',
    '-',
    '÷',
    '·',
    '×',
    '/',
    ',',
    '`',
    '(',
    ')',
    '!',
    '?',
    '~',
    ';',
    '"',
    '_',
    '∠',
    '|',
    '[',
    ']',
    '{',
    '}',
    '<',
    '>',
    '¢',
    '%',
    '&',
    '°',
    '•',
    '^',
    '⇒',
    '√',
    '≈',
    '≠',
    '≤',
    '≥',
    '○',
    '²',
    '³',
]

special_tokens = emoji_and_symbols_tokens + super_special_tokens + typographic_tokens


def special_token_to_label_mapper(special_token):
    # other labels, processed throughout the code:
    # '@extremely_common@'
    # '@very_common@'
    # '@moderately_common@'
    # '@less_common@'
    # '@rare@'
    # '@word_filler@'
    if special_token in emoji_and_symbols_tokens:
        return '@emoji_and_symbols_tokens@'
    if special_token in super_special_tokens:
        return '@super_special_tokens@'
    if special_token in typographic_tokens:
        return '@typographic_tokens@'
    if special_token.isdigit():
        return '@digit_tokens@'
    return None


def compute_pair_scores(splits):
    letter_freqs = defaultdict(int)
    pair_freqs = defaultdict(int)
    for word, _freq in words.items():
        # freq = 1 # comment in for every word to be a weight of 1 - this is divergent from wordpiece/bpe
        freq = _freq
        split = splits[word]
        if len(split) == 1:
            letter_freqs[split[0]] += freq
            continue
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            letter_freqs[split[i]] += freq
            pair_freqs[pair] += freq
        letter_freqs[split[-1]] += freq
    scores = {
        # wordpiece optimizes for smallest vocab. this is just compute efficiency. works ok for translation.
        # bpe optimizes for most frequent pairs, which is closer to human learning.
        # whatever we see most, we memorize.
        pair: freq  # / (letter_freqs[pair[0]] * letter_freqs[pair[1]])
        for pair, freq in pair_freqs.items()
    }
    return scores


def merge_pair(a, b, splits):
    for word in words:
        split = splits[word]
        if len(split) == 1:
            continue
        i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b:
                merge = a + b[2:] if b.startswith('##') else a + b
                split = split[:i] + [merge] + split[i + 2 :]
            else:
                i += 1
        splits[word] = split
    return splits


def special_token_split(s, delimiters):
    delimiters.sort(key=len, reverse=True)
    pattern = re.compile('(' + '|'.join(map(re.escape, delimiters)) + ')')
    result = []
    for part in pattern.split(s):
        if part:
            result.append(part)
    return result


def split_to_digits(s):
    result = []
    current_segment = ''
    for char in s:
        if char.isdigit():
            if current_segment:
                result.append(current_segment)
                current_segment = ''
            result.append(char)
        else:
            current_segment += char
    if current_segment:
        result.append(current_segment)
    return result


def digit_split(tokens):
    digit_pattern = re.compile(r'\d')
    result = []
    special_tokens_set = set(special_tokens)
    for token in tokens:
        is_in_special_tokens = token in special_tokens_set
        if bool(digit_pattern.search(token)) and not is_in_special_tokens:
            result.extend(split_to_digits(token))
        else:
            result.append(token)
    return result


def tokenize(text, splits, commonality_map):
    tokens = []
    labels = []
    for token in digit_split(special_token_split(text, special_tokens)):
        if token in splits:
            commonality_label = commonality_map[token]
            if commonality_label is None:
                exit(f'commonality_label is None for token {token}')
            labels.append(commonality_label)
            is_first = True
            for split_token in splits[token]:
                if not is_first:
                    labels.append('@word_filler@')
                tokens.append(split_token)
                is_first = False
        else:
            labels.append(special_token_to_label_mapper(token))
            tokens.append(token)
    return tokens, labels


def tokenize_word_map(text, splits, commonality_map):
    tokens = []
    labels = []
    all_splits_tokens = set([item for sublist in splits.values() for item in sublist])
    for token in special_token_split(text, special_tokens):
        token_with_hashes = f'##{token}'
        if token in splits:
            commonality_label = commonality_map[token]
            if commonality_label is None:
                exit(f'commonality_label is None for token {token}')
            labels.append(commonality_label)
            is_first = True
            for split_token in splits[token]:
                if not is_first:
                    labels.append('@word_filler@')
                tokens.append(split_token)
                is_first = False
        elif token_with_hashes in all_splits_tokens:
            labels.append('@word_filler@')
            tokens.append(token_with_hashes)
        else:
            labels.append(special_token_to_label_mapper(token))
            tokens.append(token)
    return tokens, labels


def tokens_to_array_of_numbers(tokens):
    full_vocab = list()
    full_vocab += digit_vocab
    full_vocab += alphabet_vocab
    full_vocab += vocab
    full_vocab += special_tokens
    full_vocab = list(dict.fromkeys(full_vocab))
    full_vocab_from_tokens = list(set(tokens))
    not_needed = set(full_vocab) - set(full_vocab_from_tokens)
    print('not_needed set (should always be empty):', not_needed)
    if len(not_needed) != 0:
        raise Exception('not_needed set is not empty')
    full_vocab_map = {
        token: index for token, index in zip(full_vocab, range(len(full_vocab)))
    }
    result = []
    for token in tokens:
        if token in full_vocab_map:
            result.append(full_vocab_map[token])
        else:
            raise Exception(f'Token {token} is not in vocab')
    return [result, full_vocab]


def labels_to_array_of_numbers(labels):
    labels_map = dict()
    counter = 0
    for label in labels:
        if label not in labels_map:
            labels_map[label] = counter
            counter += 1
    result = []
    for label in labels:
        if label is None:
            raise Exception('label is None')
        if label in labels_map:
            result.append(labels_map[label])
        else:
            raise Exception(f'Label {label} is not in all_labels_list')
    return result


def tokens_to_array_of_numbers_without_full_vocab(tokens, full_vocab):
    result = []
    for token in tokens:
        if token in full_vocab:
            result.append(full_vocab.index(token))
        else:
            raise Exception(f'Token {token} is not in vocab')
    return result


if __name__ == '__main__':
    #################### LOAD TEXT ####################
    print('loading text')
    with open('trainingdata.txt', 'r') as f:
        initial_text = f.read()

    text = initial_text
    #################### /LOAD TEXT ####################

    #################### GENERATE WORD SPLITS ####################
    print('generating word splits')
    text = text.lower()
    for special_token in special_tokens:
        text = text.replace(special_token, ' ')
    # replace numbers
    text = re.sub(r'\d+', ' ', text)

    tokens = text.split()
    token_counts = Counter(tokens)
    most_common_tokens = token_counts.most_common()

    words = {}
    for token, count in most_common_tokens:
        token = token.strip("'")
        if token not in words:
            words[token] = 0
        words[token] += count

    splits = {
        # FYI: we don't differentiate between pieces at the beginning of a word and pieces from any other part of the word.
        word: [f'##{c}' if i == 0 else f'##{c}' for i, c in enumerate(word)]
        for word in words.keys()
    }
    #################### /GENERATE WORD SPLITS ####################

    #################### BPE MERGE UP TO VOCAB SIZE ####################
    print('bpe merging up to vocab size')
    alphabet_vocab = map(lambda c: f'##{c}', list('abcdefghijklmnopqrstuvwxyz'))
    digit_vocab = list('0123456789')
    vocab = list()

    vocab_size = (
        # always 761 because we are using the labels embedding table
        761
    )  # should this be number of phonemes or syllables? thinking 44, 100 or something.
    # now going for 1024 total vocab size
    while len(vocab) < vocab_size:
        scores = compute_pair_scores(splits)
        best_pair, max_score = '', None
        for pair, score in scores.items():
            if max_score is None or max_score < score:
                best_pair = pair
                max_score = score
        splits = merge_pair(*best_pair, splits)
        new_token = (
            best_pair[0] + best_pair[1][2:]
            if best_pair[1].startswith('##')
            else best_pair[0] + best_pair[1]
        )

        # check that best_pair[0] is still in splits. remove from vocab if not.
        is_best_pair_0_removed_now = not any(
            [best_pair[0] in split for split in splits.values()]
        )
        if is_best_pair_0_removed_now:
            vocab.remove(best_pair[0])

        # check that best_pair[1] is still in splits. remove from vocab if not.
        is_best_pair_1_removed_now = not any(
            [best_pair[1] in split for split in splits.values()]
        )
        if is_best_pair_1_removed_now:
            vocab.remove(best_pair[1])

        vocab.append(new_token)

    #################### /BPE MERGE UP TO VOCAB SIZE ####################

    #################### GENERATE COMMONALITY MAP ####################
    print('generating commonality map')

    sorted_words = sorted(words, key=words.get, reverse=True)

    total_words = len(sorted_words)
    boundary_1 = total_words // 5
    boundary_2 = boundary_1 * 2
    boundary_3 = boundary_1 * 3
    boundary_4 = boundary_1 * 4
    commonality_map = {}

    for i in range(total_words):
        if i < boundary_1:
            commonality_map[sorted_words[i]] = '@extremely_common@'
        elif i < boundary_2:
            commonality_map[sorted_words[i]] = '@very_common@'
        elif i < boundary_3:
            commonality_map[sorted_words[i]] = '@moderately_common@'
        elif i < boundary_4:
            commonality_map[sorted_words[i]] = '@less_common@'
        else:
            commonality_map[sorted_words[i]] = '@rare@'

    #################### /GENERATE COMMONALITY MAP ####################

    #################### TOKENIZE WORD MAP AND THE TEXT ####################
    print('tokenizing word map and the text')
    spelling_map_text = ''
    spelling_map_text += '\<\|document\|\>letter map for words\n'
    for word in splits:
        if len(splits[word]) == 1:
            word_split_to_letters = list(word)
            spelling_map_text += f'{word}: {"-".join(word_split_to_letters)}\n'

    splits_more_than_one = [splits[word] for word in splits if len(splits[word]) > 1]
    flattened_splits_more_than_one = [
        item for sublist in splits_more_than_one for item in sublist
    ]
    uniq_flattened_splits_more_than_one = list(
        dict.fromkeys(flattened_splits_more_than_one)
    )
    for piece in uniq_flattened_splits_more_than_one:
        piece = piece[2:] if piece.startswith('##') else piece
        piece_split_to_letters = list(piece)
        spelling_map_text += f'{piece}: {"-".join(piece_split_to_letters)}\n'

    spelling_map_text += '\n\n\n'

    word_map_toks, word_map_lbls = tokenize_word_map(
        spelling_map_text, splits, commonality_map
    )
    toks, lbls = tokenize(initial_text.lower(), splits, commonality_map)

    #################### /TOKENIZE WORD MAP AND THE TEXT ####################

    #################### SAVE TOKENS AND FULL VOCAB ####################
    print('saving tokens and full vocab')
    tokens, full_vocab = tokens_to_array_of_numbers(word_map_toks + toks)
    labels = labels_to_array_of_numbers(word_map_lbls + lbls)

    suffix = '-label_embeddings'

    with open(f'tokens{suffix}.json', 'w') as f:
        json.dump(tokens, f)

    with open(f'labels{suffix}.json', 'w') as f:
        json.dump(labels, f)

    labels_map = {
        label_non_vectorized: label
        for label, label_non_vectorized in zip(labels, word_map_lbls + lbls)
    }
    with open(f'labels_map{suffix}.json', 'w') as f:
        json.dump(labels_map, f, indent=4)

    set_toks = set(word_map_toks + toks)
    set_toks_without_special_tokens = set_toks - set(special_tokens)
    set_toks_without_special_tokens_and_vocab = (
        set_toks_without_special_tokens
        - set(vocab)
        - set(digit_vocab)
        - set(alphabet_vocab)
    )
    print('set_toks (vocab_size)', len(set_toks))
    sorted_set_toks_without_special_tokens_and_vocab = sorted(
        set_toks_without_special_tokens_and_vocab
    )

    # FYI: one is used for decode and one is used for encode. can probably refactor to use the same.
    with open(f'full_vocab{suffix}.json', 'w') as f:
        json.dump(full_vocab, f)

    with open(f'splits{suffix}.json', 'w') as f:
        json.dump(splits, f)

    with open(f'commonality_map{suffix}.json', 'w') as f:
        json.dump(commonality_map, f)

    #################### /SAVE TOKENS AND FULL VOCAB ####################
