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
    '⋅', # TODO: same as ⋅
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
    '½', # TODO: should be 1/2
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
    '−', # TODO: should be -
    '²',
    '³',
    '✓',
    '✔', # TODO: same as ✓
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

vocab_size = 754 # should this be number of phonemes or syllables? thinking 44, 100 or something.
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
    # TODO: I'm not sure why not_needed is actually non-empty. Should be always empty - somehow some BPE tokens are never used.
    not_needed = set(full_vocab) - set(full_vocab_from_tokens)
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

def decode(tokens):
    tokens = [full_vocab[token] for token in tokens]
    tokens = [token[2:] if token.startswith("##") else token for token in tokens]
    return ''.join(tokens)

model_output = [0, 219, 543, 218, 224, 224, 222, 222, 219, 302, 678, 219, 284, 219, 503, 690, 18, 224, 224, 445, 219, 315, 219, 472, 219, 854, 219, 634, 500, 219, 366, 219, 315, 219, 536, 567, 219, 14, 220, 602, 219, 366, 219, 325, 219, 408, 225, 219, 27, 219, 839, 219, 295, 219, 220, 936, 218, 219, 490, 219, 325, 219, 381, 219, 284, 219, 588, 219, 557, 232, 219, 325, 219, 401, 689, 219, 18, 417, 398, 219, 290, 219, 559, 219, 451, 219, 402, 219, 588, 219, 793, 313, 218, 224, 224, 217, 217, 323, 24, 219, 312, 350, 217, 217, 219, 221, 219, 385, 219, 312, 307, 219, 286, 219, 284, 219, 602, 219, 718, 285, 968, 219, 402, 219, 936, 218, 219, 221, 219, 490, 219, 284, 219, 936, 219, 557, 219, 506, 291, 219, 598, 219, 286, 219, 284, 219, 602, 232, 219, 445, 219, 315, 219, 32, 219, 863, 219, 603, 219, 386, 218, 224, 224, 332, 566, 232, 219, 332, 566, 232, 219, 332, 566, 236, 219, 234, 27, 28, 219, 226, 219, 27, 30, 219, 223, 219, 28, 26, 235, 218, 219, 345, 232, 219, 284, 219, 1008, 219, 294, 219, 28, 219, 226, 219, 31, 219, 226, 219, 31, 219, 223, 219, 29, 26, 219, 867, 18, 218, 224, 224, 222, 222, 222, 219, 302, 19, 348, 2, 219, 673, 224, 224, 284, 219, 302, 318, 282, 219, 294, 219, 284, 219, 12, 567, 3, 288, 219, 532, 428, 219, 283, 219, 0, 219, 969, 219, 306, 219, 312, 350, 219, 366, 219, 294, 219, 866, 219, 283, 219, 301, 859, 219, 295, 219, 787, 924, 343, 218, 219, 409, 219, 300, 219, 315, 219, 381, 293, 219, 407, 219, 28, 219, 292, 219, 385, 219, 610, 232, 219, 300, 219, 378, 219, 998, 219, 374, 546, 219, 284, 219, 312, 307, 219, 456, 584, 218, 224, 224, 217, 217, 311, 281, 369, 219, 27, 217, 217, 225, 219, 449, 219, 402, 219, 612, 219, 505, 219, 312, 350, 218, 219, 310, 219, 378, 219, 330, 219, 612, 398, 219, 318, 633, 343, 219, 407, 219, 829, 219, 28, 219, 292, 219, 31, 218, 219, 411, 219, 312, 307, 219, 695, 219, 562, 219, 407, 219, 30, 237, 224, 224, 217, 217, 282, 322, 219, 27, 217, 217, 225, 219, 28, 232, 219, 456, 220, 18, 219, 612, 219, 312, 350, 218, 219, 310, 220, 18, 219, 30, 219, 292, 219, 27, 26, 225, 219, 290, 219, 330, 219, 318, 633, 343, 219, 407, 219, 28, 219, 347, 386, 219, 0, 219, 285, 854, 370, 218, 224, 224, 222, 222, 219, 311, 352, 334, 219, 28, 224, 224, 456, 220, 18, 219, 970, 219, 325, 219, 27, 28, 219, 388, 470, 218, 219, 325, 219, 506, 219, 451, 219, 385, 219, 306, 219, 284, 219, 928, 959, 219, 290, 219, 697, 219, 400, 219, 411, 219, 312, 350, 219, 957, 10, 296, 219, 290, 219, 449, 218, 224, 224, 456, 220, 18, 219, 561, 219, 440, 219, 579, 399, 219, 292, 219, 346, 370, 464, 219, 411, 219, 385, 219, 294, 219, 292, 219, 615, 219, 290, 219, 683, 219, 411, 219, 385, 219, 604, 864, 219, 681, 219, 629, 395, 219, 432, 218, 224, 224, 217, 217, 311, 281, 369, 219, 27, 225, 217, 217, 219, 733, 14, 309, 219, 284, 219, 859, 219, 625, 986, 219, 298, 219, 284, 219, 330, 784, 219, 300, 219, 22, 379, 218, 219, 645, 232, 219, 594, 219, 342, 219, 393, 219, 565, 281, 219, 625, 986, 219, 792, 219, 471, 219, 432, 219, 34, 219, 984, 219, 286, 219, 284, 219, 5, 349, 295, 218, 224, 224, 217, 217, 311, 281, 369, 219, 28, 225, 217, 217, 219, 378, 219, 300, 219, 562, 219, 0, 219, 503, 6, 219, 306, 219, 625, 986, 219, 347, 219, 469, 293, 219, 31, 219, 5, 11, 384, 219, 409, 219, 792, 219, 536, 22, 219, 290, 219, 469, 219, 0, 219, 29, 219, 505, 219, 625, 986, 219, 479, 219, 284, 219, 503, 6, 218, 219, 283, 219, 384, 219, 503, 10, 580, 232, 219, 325, 219, 315, 219, 285, 301, 398, 219, 577, 293, 219, 283, 219, 284, 219, 503, 690, 219, 610, 221, 407, 219, 610, 218, 224, 224, 284, 219, 655, 302, 219, 283, 219, 346, 370, 464, 293, 219, 312, 350, 219, 294, 219, 816, 219, 284, 219, 240, 886, 281, 964, 232, 240, 219, 283, 219, 284, 219, 285, 583, 303, 478, 8, 15, 219, 330, 799, 219, 472, 219, 312, 350, 219, 347, 219, 28, 219, 292, 219, 29, 26, 232, 219, 284, 219, 691, 626, 0, 473, 219, 292, 219, 284, 219, 328, 593, 8, 511, 473, 219, 294, 219, 32, 218, 224, 224, 217, 217, 311, 352, 334, 219, 28, 225, 217, 217, 219, 0, 219, 361, 335, 219, 306, 219, 28, 219, 23, 219, 31, 219, 223, 219, 27, 231, 28, 219, 234, 474, 289, 614, 219, 407, 219, 30, 235, 219, 292, 219, 310, 219, 407, 219, 29, 219, 234, 32, 235, 219, 23, 219, 29, 219, 223, 219, 28, 235, 224, 224, 423, 219, 366, 219, 300, 220, 285, 219, 286, 219, 362, 770, 293, 219, 284, 219, 311, 281, 354, 291, 232, 219, 300, 219, 367, 219, 423, 219, 853, 219, 290, 219, 401, 321, 219, 286, 219, 290, 219, 284, 219, 503, 690, 219, 726, 291, 232, 219, 285, 517, 307, 219, 366, 219, 368, 605, 293, 219, 510, 219, 437, 938, 219, 315, 219, 729, 383, 218, 219, 736, 219, 358, 219, 1, 524, 232, 219, 292, 219, 975, 220, 19, 219, 651, 321, 219, 432, 219, 409, 219, 300, 219, 615, 219, 533, 219, 0, 219, 1, 310, 219, 653, 236, 219, 542, 219, 310, 219, 386, 232, 219, 292, 219, 300, 220, 337, 219, 301, 806, 219, 845, 219, 0, 219, 820, 219, 295, 219, 323, 4, 1009, 219, 564, 219, 315, 219, 282, 219, 344, 318, 303, 219]
print(decode(model_output))