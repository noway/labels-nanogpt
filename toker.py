from collections import Counter
import re
import pyphen


with open('trainingdata.txt', 'r') as f:
    text = f.read()
special_tokens = ["<|document|>"]
common_words = {"the", "and", "is", "in", "on", "at", "of", "------------------------------------------------------------------------"}

# Token patterns
token_patterns = {
    "special_token": re.compile(r"^<\|document\|>"),
    "space": re.compile(r"^\s+"),
    "number": re.compile(r"^\d+"),
    "dot": re.compile(r"^\."),
    "equals": re.compile(r"^="),
    "minus": re.compile(r"^-"),
    "plus": re.compile(r"^\+"),
    "comma": re.compile(r"^,"),
    "colon": re.compile(r"^:"),
    "underscore_slash": re.compile(r"^_/"),
    "double_asterisk": re.compile(r"^\*\*"),
    "word": re.compile(r"^(?:{})\b".format("|".join(re.escape(word) for word in common_words)))
}

def match_token(text, cursor):
    for token_type, pattern in token_patterns.items():
        match = pattern.match(text[cursor:cursor+10])
        if match:
            return match.group(), cursor + len(match.group())
    return text[cursor:cursor+2], cursor + 2

def custom_tokenize(text):
    tokens = []
    cursor = 0

    while cursor < len(text):
        token, cursor = match_token(text, cursor)
        if token:
            tokens.append(token)

    return tokens


# tokens = custom_tokenize(text)
# lowercase
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
        if syllable not in all_syllables:
            all_syllables[syllable] = 0
        all_syllables[syllable] += count

# all_syllables_items = all_syllables
# print (all_syllables_items)
# sorted_all_syllables_items = sorted(all_syllables_items, key=lambda x: len(x[0]), reverse=True)
# for syllable, count in sorted_all_syllables_items:
#     print (syllable, count)

# sorted_all_syllables_items to dict
# sorted_all_syllables_items_dict = {}
# for syllable, count in sorted_all_syllables_items:
#     sorted_all_syllables_items_dict[syllable] = count

splits = {
    word: [c if i == 0 else f"##{c}" for i, c in enumerate(word)]
    for word in all_syllables.keys()
}

# print(sorted_all_syllables_items)
print(splits)

# def compute_pair_scores()

syllables_joined = sorted(set(''.join(all_syllables.keys())))
print (syllables_joined)