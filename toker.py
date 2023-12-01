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

def decode(tokens):
    tokens = [full_vocab[token] for token in tokens]
    tokens = [token[2:] if token.startswith("##") else token for token in tokens]
    return ''.join(tokens)

model_output = [0, 509, 219, 306, 219, 373, 219, 378, 219, 306, 219, 543, 219, 292, 219, 296, 9, 14, 24, 843, 218, 219, 300, 219, 423, 219, 290, 219, 362, 1, 497, 219, 472, 219, 309, 335, 219, 306, 219, 411, 219, 300, 219, 373, 219, 294, 236, 224, 224, 222, 222, 222, 219, 353, 742, 219, 27, 225, 219, 312, 307, 219, 979, 293, 219, 621, 520, 224, 224, 217, 217, 342, 219, 290, 219, 670, 225, 217, 217, 219, 221, 219, 15, 356, 887, 309, 219, 347, 219, 670, 429, 219, 366, 219, 283, 534, 468, 331, 303, 219, 234, 330, 748, 235, 219, 451, 219, 27, 218, 219, 594, 219, 284, 219, 312, 307, 219, 306, 219, 525, 219, 286, 219, 0, 219, 313, 341, 555, 219, 295, 219, 534, 285, 719, 286, 431, 218, 224, 224, 217, 217, 411, 219, 290, 219, 413, 225, 217, 217, 219, 577, 219, 347, 219, 284, 219, 557, 232, 219, 472, 219, 313, 341, 555, 219, 505, 219, 564, 219, 315, 219, 284, 219, 359, 19, 983, 218, 217, 219, 221, 219, 901, 219, 300, 219, 449, 219, 366, 219, 373, 219, 295, 219, 422, 742, 219, 451, 219, 27, 219, 290, 219, 28, 26, 219, 351, 219, 0, 219, 692, 281, 219, 312, 307, 218, 219, 221, 219, 440, 219, 294, 219, 490, 219, 300, 219, 367, 219, 422, 549, 219, 290, 219, 596, 219, 342, 219, 393, 219, 772, 219, 367, 219, 490, 219, 300, 219, 367, 232, 219, 300, 219, 315, 219, 999, 219, 0, 219, 30, 219, 19, 285, 298, 18, 218, 224, 224, 456, 219, 558, 219, 284, 219, 312, 307, 219, 311, 352, 334, 219, 306, 219, 751, 219, 292, 219, 300, 219, 697, 18, 219, 300, 219, 411, 219, 564, 219, 506, 219, 315, 225, 224, 224, 221, 219, 219, 219, 312, 307, 219, 306, 219, 914, 225, 219, 28, 219, 914, 224, 221, 219, 219, 219, 300, 219, 367, 225, 219, 34, 224, 224, 217, 217, 353, 742, 219, 31, 225, 217, 217, 219, 411, 219, 467, 647, 219, 413, 219, 300, 219, 325, 219, 367, 237, 219, 627, 319, 219, 325, 219, 615, 219, 290, 219, 562, 219, 57, 27, 26, 219, 603, 232, 219, 300, 219, 482, 219, 367, 219, 654, 219, 506, 293, 219, 372, 219, 284, 219, 509, 219, 290, 219, 284, 219, 604, 740, 219, 559, 219, 382, 218, 224, 224, 217, 217, 311, 281, 369, 219, 34, 225, 217, 217, 219, 627, 319, 219, 1, 956, 10, 793, 313, 232, 219, 8, 4, 344, 431, 232, 219, 1, 956, 10, 793, 313, 232, 219, 683, 293, 219, 342, 219, 775, 219, 982, 219, 482, 219, 300, 219, 559, 219, 762, 218, 224, 224, 217, 217, 311, 352, 334, 219, 282, 322, 225, 217, 217, 219, 300, 219, 367, 219, 30, 219, 459, 835, 219, 351, 219, 0, 219, 285, 329, 3, 218, 219, 436, 281, 219, 615, 18, 219, 29, 219, 4, 6, 6, 6, 537, 2, 219, 738, 219, 31, 219, 505, 219, 459, 835, 219, 300, 219, 615, 219, 290, 219, 562, 218, 219, 409, 219, 8, 219, 4, 12, 460, 358, 219, 459, 613, 219, 999, 18, 219, 57, 27, 26, 232, 219, 342, 219, 775, 219, 482, 219, 403, 219, 459, 613, 219, 482, 219, 4, 0, 418, 4, 219, 301, 398, 219, 643, 417, 343, 219, 283, 219, 290, 529, 237, 224, 224, 222, 222, 222, 222, 219, 361, 962, 224, 224, 217, 217, 311, 352, 334, 225, 217, 217, 219, 627, 319, 219, 300, 219, 367, 219, 948, 219, 356, 219, 30, 219, 612, 219, 292, 219, 300, 219, 651, 321, 219, 30, 219, 656, 218, 219, 342, 219, 324, 23, 219, 520, 219, 840, 219, 57, 29, 218, 219, 409, 219, 300, 219, 309, 337, 219, 635, 219, 818, 429, 219, 7, 294, 219, 18, 13, 566, 18, 219, 347, 219, 0, 219, 285, 854, 370, 219, 28, 232, 219, 515, 219, 294, 219, 29, 219, 818, 429, 219, 603, 219, 627, 319, 219, 30, 219, 226, 219, 30, 219, 226, 219, 30, 219, 223, 219, 27, 26, 218, 224, 224, 222, 222, 222, 219, 29, 218, 219, 488, 219, 730, 289, 657, 293, 219, 435, 224, 224, 435, 219, 315, 219, 436, 580, 738, 236, 219, 300, 219, 367, 219, 330, 296, 219, 358, 219, 509, 219, 290, 219, 346, 370, 665, 872, 218, 219, 309, 299, 327, 219, 409, 219, 300, 219, 367, 219, 0, 219, 418, 4, 333, 219, 306, 219, 547, 430, 232, 219, 339, 1017, 219, 283, 637, 232, 219, 295, 219, 611, 8, 485, 361, 219, 284, 219, 282, 581, 219, 400, 293, 219, 292, 219, 17, 506, 218, 224, 224, 217, 217, 217, 311, 281, 369, 225, 217, 217, 219, 300, 219, 378, 219, 449, 219, 0, 219, 582, 219, 965, 286, 219, 385, 219, 547, 430, 219, 292, 219, 0, 219, 2, 317, 10, 219, 290, 219, 0, 219, 547, 430, 218, 219, 411, 219, 414, 219, 294, 219, 358, 219, 330, 3, 236, 224, 224, 217, 217, 311, 352, 334, 225, 217, 217, 219, 221, 219, 433, 442, 219, 358, 219, 494, 10, 219, 294, 219, 28, 219, 882, 219, 540, 232, 219, 638, 313, 219, 292, 219, 0, 219, 608, 24, 218, 219, 221, 219, 467, 313, 463, 519, 219, 676, 219, 793, 1010, 219, 8, 398, 218, 224, 224, 217, 217, 282, 455, 225, 217, 217, 219, 221, 219, 604, 740, 219, 881, 219, 469, 219, 358, 219, 849, 219, 292, 219, 412, 934, 219, 629, 978, 291, 218, 219, 221, 219, 37, 224, 224, 36, 224, 224, 222, 222, 222, 219, 31, 218, 219, 466, 319, 225, 219, 283, 715, 693, 303, 219, 292, 219, 290, 219, 877, 224, 224, 217, 217, 217, 709, 219, 453, 674, 19, 225, 217, 217, 219, 300, 219, 346, 370, 464, 219, 402, 219, 285, 583, 303, 219, 306, 219, 676, 219, 767, 232, 219, 399, 660, 232, 219, 292, 219, 877, 293]
print(decode(model_output))