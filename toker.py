from collections import Counter
import re
import pyphen
from collections import defaultdict

with open('trainingdata.txt', 'r') as f:
    text = f.read()

text = text.lower()
text = text.replace('*️⃣', ' ')
text = text.replace('1️⃣', ' ')
text = text.replace('2️⃣', ' ')
text = text.replace('3️⃣', ' ')
text = text.replace('4️⃣', ' ')
text = text.replace('5️⃣', ' ')
text = text.replace('6️⃣', ' ')
text = text.replace('7️⃣', ' ')
text = text.replace('8️⃣', ' ')
text = text.replace('9️⃣', ' ')
text = text.replace('🔟', ' ')
text = text.replace('⚪', ' ')
text = text.replace('⚫', ' ')
text = text.replace('⚽', ' ')
text = text.replace('⚾', ' ')
text = text.replace('✂', ' ')
text = text.replace('✅', ' ')
text = text.replace('✈', ' ')
text = text.replace('✋', ' ')
text = text.replace('✏', ' ')
text = text.replace('✦', ' ')
text = text.replace('✨', ' ')
text = text.replace('✪', ' ')
text = text.replace('❄', ' ')
text = text.replace('❌', ' ')
text = text.replace('❏', ' ')
text = text.replace('❤', ' ')
text = text.replace('➡', ' ')
text = text.replace('⬅', ' ')
text = text.replace('⬜', ' ')
text = text.replace('⬡', ' ')
text = text.replace('⭐', ' ')
text = text.replace('️', ' ')
text = text.replace('🌟', ' ')
text = text.replace('🌱', ' ')
text = text.replace('🌲', ' ')
text = text.replace('🌳', ' ')
text = text.replace('🌴', ' ')
text = text.replace('🌵', ' ')
text = text.replace('🌷', ' ')
text = text.replace('🌸', ' ')
text = text.replace('🌹', ' ')
text = text.replace('🌺', ' ')
text = text.replace('🌻', ' ')
text = text.replace('🌼', ' ')
text = text.replace('🌾', ' ')
text = text.replace('🍀', ' ')
text = text.replace('🍂', ' ')
text = text.replace('🍃', ' ')
text = text.replace('🍇', ' ')
text = text.replace('🍉', ' ')
text = text.replace('🍊', ' ')
text = text.replace('🍌', ' ')
text = text.replace('🍎', ' ')
text = text.replace('🍏', ' ')
text = text.replace('🍐', ' ')
text = text.replace('🍒', ' ')
text = text.replace('🍓', ' ')
text = text.replace('🍕', ' ')
text = text.replace('🍞', ' ')
text = text.replace('🍦', ' ')
text = text.replace('🍩', ' ')
text = text.replace('🍪', ' ')
text = text.replace('🍬', ' ')
text = text.replace('🍳', ' ')
text = text.replace('🍴', ' ')
text = text.replace('🍽', ' ')
text = text.replace('🎈', ' ')
text = text.replace('🎉', ' ')
text = text.replace('🎒', ' ')
text = text.replace('🎨', ' ')
text = text.replace('🎵', ' ')
text = text.replace('🎸', ' ')
text = text.replace('🏀', ' ')
text = text.replace('🏎', ' ')
text = text.replace('🏘', ' ')
text = text.replace('🏠', ' ')
text = text.replace('🏡', ' ')
text = text.replace('🏢', ' ')
text = text.replace('🏰', ' ')
text = text.replace('🐁', ' ')
text = text.replace('🐈', ' ')
text = text.replace('🐌', ' ')
text = text.replace('🐘', ' ')
text = text.replace('🐙', ' ')
text = text.replace('🐝', ' ')
text = text.replace('🐟', ' ')
text = text.replace('🐠', ' ')
text = text.replace('🐤', ' ')
text = text.replace('🐦', ' ')
text = text.replace('🐭', ' ')
text = text.replace('🐰', ' ')
text = text.replace('🐱', ' ')
text = text.replace('🐳', ' ')
text = text.replace('🐵', ' ')
text = text.replace('🐶', ' ')
text = text.replace('🐷', ' ')
text = text.replace('🐸', ' ')
text = text.replace('🐻', ' ')
text = text.replace('👉', ' ')
text = text.replace('👟', ' ')
text = text.replace('👧', ' ')
text = text.replace('👨', ' ')
text = text.replace('👩', ' ')
text = text.replace('💦', ' ')
text = text.replace('💧', ' ')
text = text.replace('💼', ' ')
text = text.replace('📏', ' ')
text = text.replace('📖', ' ')
text = text.replace('📘', ' ')
text = text.replace('📚', ' ')
text = text.replace('🔍', ' ')
text = text.replace('🔎', ' ')
text = text.replace('🔟', ' ')
text = text.replace('🔢', ' ')
text = text.replace('🔥', ' ')
text = text.replace('🔮', ' ')
text = text.replace('🔴', ' ')
text = text.replace('🔵', ' ')
text = text.replace('🔶', ' ')
text = text.replace('🔺', ' ')
text = text.replace('🔼', ' ')
text = text.replace('🕊', ' ')
text = text.replace('🕓', ' ')
text = text.replace('🖍', ' ')
text = text.replace('🖐', ' ')
text = text.replace('😄', ' ')
text = text.replace('😊', ' ')
text = text.replace('🚀', ' ')
text = text.replace('🚌', ' ')
text = text.replace('🚒', ' ')
text = text.replace('🚓', ' ')
text = text.replace('🚕', ' ')
text = text.replace('🚗', ' ')
text = text.replace('🚙', ' ')
text = text.replace('🚛', ' ')
text = text.replace('🚜', ' ')
text = text.replace('🚧', ' ')
text = text.replace('🚲', ' ')
text = text.replace('🛑', ' ')
text = text.replace('🛴', ' ')
text = text.replace('🟠', ' ')
text = text.replace('🟡', ' ')
text = text.replace('🟢', ' ')
text = text.replace('🟣', ' ')
text = text.replace('🟥', ' ')
text = text.replace('🟦', ' ')
text = text.replace('🟨', ' ')
text = text.replace('🟩', ' ')
text = text.replace('🤚', ' ')
text = text.replace('🥕', ' ')
text = text.replace('🥚', ' ')
text = text.replace('🥣', ' ')
text = text.replace('🥤', ' ')
text = text.replace('🦆', ' ')
text = text.replace('🦩', ' ')
text = text.replace('🧀', ' ')
text = text.replace('🧐', ' ')
text = text.replace('🧒', ' ')
text = text.replace('🧦', ' ')
text = text.replace('🧸', ' ')
text = text.replace('🪁', ' ')
text = text.replace('●', ' ')
text = text.replace('☀', ' ')
text = text.replace('★', ' ')
text = text.replace('☑', ' ')
text = text.replace('\u200d', ' ')
text = text.replace('\\<\\|image\\|\\>', ' ')
text = text.replace('\\<\\|document\\|\\>', ' ')
text = text.replace('\\<\\|unsolvedproblem\\|\\>', ' ')
text = re.sub(r'\d+', ' ', text)
text = text.replace('*', ' ')
text = text.replace('.', ' ')
text = text.replace('-', ' ')
text = text.replace('#', ' ')
text = text.replace('=', ' ')
text = text.replace('\\\n', ' ')
text = text.replace('\\ ', ' ')
text = text.replace('\\_', ' ')
text = text.replace('\\]', ' ')
text = text.replace('\\[', ' ')
text = text.replace('\\[', ' ')
text = text.replace('\\^', ' ')
text = text.replace('\\|', ' ')
text = text.replace('\\/', ' ')
text = text.replace('\\$', ' ')
text = text.replace('\\<', ' ')
text = text.replace('\\>', ' ')
text = text.replace(':', ' ')
text = text.replace('+', ' ')
text = text.replace('-', ' ')
text = text.replace('÷', ' ')
text = text.replace('·', ' ')
text = text.replace('⋅', ' ') # TODO: same as ⋅
text = text.replace('×', ' ')
text = text.replace('/', ' ')
text = text.replace(',', ' ')
text = text.replace('`', ' ')
text = text.replace('(', ' ')
text = text.replace(')', ' ')
text = text.replace('!', ' ')
text = text.replace('?', ' ')
text = text.replace('~', ' ')
text = text.replace(';', ' ')
text = text.replace('"', ' ')
text = text.replace('_', ' ')
text = text.replace('∠', ' ')
text = text.replace('|', ' ')
text = text.replace('[', ' ')
text = text.replace(']', ' ')
text = text.replace('{', ' ')
text = text.replace('}', ' ')
text = text.replace('<', ' ')
text = text.replace('>', ' ')
text = text.replace('π', ' ')
text = text.replace('%', ' ')
text = text.replace('&', ' ')
text = text.replace('¢', ' ')
text = text.replace('°', ' ')
text = text.replace('•', ' ')
text = text.replace('^', ' ')
text = text.replace('\\', ' ')
text = text.replace('½', ' ') # TODO: should be 1/2
text = text.replace('¼', ' ') # TODO: should be 1/4
text = text.replace('¾', ' ') # TODO: should be 3/4
text = text.replace('⅓', ' ') # TODO: should be 1/3
text = text.replace('↑', ' ')
text = text.replace('→', ' ')
text = text.replace('↓', ' ')
text = text.replace('⇒', ' ')
text = text.replace('√', ' ')
text = text.replace('≈', ' ')
text = text.replace('≠', ' ')
text = text.replace('≤', ' ')
text = text.replace('≥', ' ')
text = text.replace('□', ' ')
text = text.replace('▢', ' ')
text = text.replace('△', ' ')
text = text.replace('○', ' ')
text = text.replace('−', ' ') # TODO: should be -
text = text.replace('²', ' ')
text = text.replace('³', ' ')
text = text.replace('✓', ' ')
text = text.replace('✔', ' ') # TODO: same as ✓

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

print (all_syllables)

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

print(splits)
print(vocab)

# splits = merge_pair(best_pair[0], best_pair[1], splits)
# print(splits)
# splits["about"]


# print(compute_pair_scores(splits))
# print(sorted_all_syllables_items)
# print(splits)

# def compute_pair_scores()

# syllables_joined = sorted(set(''.join(all_syllables.keys())))
# print (syllables_joined)