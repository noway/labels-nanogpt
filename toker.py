from collections import Counter
import re
import pyphen
from collections import defaultdict

with open('trainingdata.txt', 'r') as f:
    initial_text = f.read()

print(len(initial_text))
text = initial_text

special_tokens = [
    '------------------------------------------------------------------------',
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
    '️',
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
    '🔟',
    '🔢',
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
    '\u200d',
    '\xa0',
    '\\<\\|image\\|\\>',
    '\\<\\|document\\|\\>',
    '\\<\\|unsolvedproblem\\|\\>',
    '*',
    '.',
    ' ',
    "'",
    '-',
    '#',
    '=',
    '\n',
    '\\\n',
    '\\ ',
    '\\_',
    '\\]',
    '\\[',
    '\\[',
    '\\^',
    '\\|',
    '\\/',
    '\\$',
    '\\<',
    '\\>',
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
    'π',
    '%',
    '&',
    '¢',
    '°',
    '•',
    '^',
    '\\',
    '¼', # TODO: should be 1/4
    '¾', # TODO: should be 3/4
    '⅓', # TODO: should be 1/3
    '↑',
    '→',
    '↓',
    '⇒',
    '√',
    '≈',
    '≠',
    '≤',
    '≥',
    '□',
    '▢',
    '△',
    '○',
    '²',
    '³',
    '✓',
]
print(len(special_tokens))

text = text.lower()
for special_token in special_tokens:
    text = text.replace(special_token, ' ')
text = re.sub(r'\d+', ' ', text)

# replace numbers

tokens = text.split()
token_counts = Counter(tokens)
most_common_tokens = token_counts.most_common()
dic = pyphen.Pyphen(lang='en_US')

all_syllables = {}
for token, count in most_common_tokens:
    token = token.strip("'")
    syllables = dic.inserted(token)
    syllables = "'".join(syllables.split('-')).split("'")
    for syllable in syllables:
        if len(syllable) == 0:
            continue
        if syllable not in all_syllables:
            all_syllables[syllable] = 0
        all_syllables[syllable] += count

# print (all_syllables)

splits = {
    # FYI: we don't differentiate between pieces at the beginning of a word and pieces from any other part of the word.
    word: [ f"##{c}" if i == 0 else f"##{c}" for i, c in enumerate(word)]
    for word in all_syllables.keys()
}

def compute_pair_scores(splits):
    letter_freqs = defaultdict(int)
    pair_freqs = defaultdict(int)
    for word, _freq in all_syllables.items():
        # freq = 1 # every word has a weight of 1 - this is divergent from wordpiece/bpe
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
        pair: freq # / (letter_freqs[pair[0]] * letter_freqs[pair[1]])
        for pair, freq in pair_freqs.items()
    }
    return scores

def merge_pair(a, b, splits):
    for word in all_syllables:
        split = splits[word]
        if len(split) == 1:
            continue
        i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b:
                merge = a + b[2:] if b.startswith("##") else a + b
                split = split[:i] + [merge] + split[i + 2 :]
            else:
                i += 1
        splits[word] = split
    return splits

alphabet_vocab = map(lambda c: f"##{c}", list("abcdefghijklmnopqrstuvwxyz"))
digit_vocab = list("0123456789")
vocab = list()

vocab_size = 744 # should this be number of phonemes or syllables? thinking 44, 100 or something.
# now going for 1024 total vocab size
while len(vocab) < vocab_size:
    scores = compute_pair_scores(splits)
    best_pair, max_score = "", None
    for pair, score in scores.items():
        if max_score is None or max_score < score:
            best_pair = pair
            max_score = score
    splits = merge_pair(*best_pair, splits)
    new_token = (
        best_pair[0] + best_pair[1][2:]
        if best_pair[1].startswith("##")
        else best_pair[0] + best_pair[1]
    )

    # check that best_pair[0] is still in splits. remove from vocab if not.
    is_best_pair_0_removed_now = not any([best_pair[0] in split for split in splits.values()])
    if is_best_pair_0_removed_now:
        vocab.remove(best_pair[0])
    # check that best_pair[1] is still in splits. remove from vocab if not.
    is_best_pair_1_removed_now = not any([best_pair[1] in split for split in splits.values()])
    if is_best_pair_1_removed_now:
        vocab.remove(best_pair[1])

    vocab.append(new_token)

print(splits)
print(len(vocab))


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
    for token in tokens:
        is_in_special_tokens = token in special_tokens
        if bool(digit_pattern.search(token)) and not is_in_special_tokens:
            result.extend(split_to_digits(token))
        else:
            result.append(token)
    return result

def syllable_split(tokens):
    result = []
    for token in tokens:
        if token.isalpha():
            syllables = dic.inserted(token)
            syllables = "'".join(syllables.split('-')).split("'")
            for syllable in syllables:
                if len(syllable) == 0:
                    continue
                result.append(syllable)
        else:
            result.append(token)
    return result


def tokenize(text, splits):
    tokens = []
    for token in syllable_split(digit_split(special_token_split(text, special_tokens))):
        # print (token)
        if token in splits:
            tokens.extend(splits[token])
        else:
            tokens.append(token)
    return tokens

toks = tokenize(initial_text.lower(), splits)

def tokens_to_array_of_numbers(tokens):
    full_vocab = list()
    full_vocab += alphabet_vocab
    full_vocab += digit_vocab
    full_vocab += special_tokens
    full_vocab += vocab
    full_vocab = list(dict.fromkeys(full_vocab))
    full_vocab_from_tokens = list(set(tokens))
    # FYI: not_needed must always be empty
    not_needed = set(full_vocab) - set(full_vocab_from_tokens)
    print('not_needed set:', not_needed)
    full_vocab = [token for token in full_vocab if token not in not_needed]
    print(len(full_vocab))
    print (full_vocab)
    result = []
    for token in tokens:
        if token in full_vocab:
            result.append(full_vocab.index(token))
        else:
            raise Exception(f"Token {token} not in vocab")
    return [result, full_vocab]

tokens, full_vocab = tokens_to_array_of_numbers(toks)

import json
with open('tokens.json', 'w') as f:
    json.dump(tokens, f)

set_toks = set(toks)
set_toks_without_special_tokens = set_toks - set(special_tokens)
set_toks_without_special_tokens_and_vocab = set_toks_without_special_tokens - set(vocab) - set(digit_vocab) - set(alphabet_vocab)
print ("set_toks", len(set_toks))
sorted_set_toks_without_special_tokens_and_vocab = sorted(set_toks_without_special_tokens_and_vocab)
print (sorted_set_toks_without_special_tokens_and_vocab, len(sorted_set_toks_without_special_tokens_and_vocab))
print(len(initial_text))
print(len(toks))

with open('full_vocab.json', 'w') as f:
    json.dump(full_vocab, f)
