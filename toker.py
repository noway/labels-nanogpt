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
    '*️⃣',
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
text = re.sub(r'\d+', ' ', text)
for special_token in special_tokens:
    text = text.replace(special_token, ' ')

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

vocab = list()

vocab_size = 100
while len(vocab) < vocab_size:
    scores = compute_pair_scores(splits)
    best_pair, max_score = "", None
    for pair, score in scores.items():
        if max_score is None or max_score < score:
            best_pair = pair
            max_score = score
    splits = merge_pair(*best_pair, splits)
    # print(splits)
    new_token = (
        best_pair[0] + best_pair[1][2:]
        if best_pair[1].startswith("##")
        else best_pair[0] + best_pair[1]
    )
    vocab.append(new_token)

# print(splits)
print(len(vocab))


def special_token_split(s, delimiters):
    delimiters.sort(key=len, reverse=True)
    pattern = re.compile('(' + '|'.join(map(re.escape, delimiters)) + ')')
    result = []
    for part in pattern.split(s):
        if part:
            result.append(part)
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
    for token in syllable_split(special_token_split(text, special_tokens)):
        # print (token)
        if token in splits:
            tokens.extend(splits[token])
        else:
            tokens.append(token)
    return tokens

toks = tokenize(initial_text.lower(), splits)
# print (toks)
set_toks = set(toks)
set_toks_without_special_tokens = set_toks - set(special_tokens)
set_toks_without_special_tokens_and_vocab = set_toks_without_special_tokens - set(vocab)
print (len(set_toks))
print (set_toks_without_special_tokens_and_vocab)
print(len(initial_text))
print(len(toks))

# splits = merge_pair(best_pair[0], best_pair[1], splits)
# print(splits)
# splits["about"]


# print(compute_pair_scores(splits))
# print(sorted_all_syllables_items)
# print(splits)

# def compute_pair_scores()

# syllables_joined = sorted(set(''.join(all_syllables.keys())))
# print (syllables_joined)