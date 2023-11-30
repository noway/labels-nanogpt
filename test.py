import re

def special_token_split(s, delimiters):
    delimiters.sort(key=len, reverse=True)
    pattern = re.compile('(' + '|'.join(map(re.escape, delimiters)) + ')')
    result = []
    for part in pattern.split(s):
        if part:
            result.append(part)
    return result

test_string = "Hello, world! This is a test-string."
delimiters = [',', ' ', '-', '!', 'lo']
result = special_token_split(test_string, delimiters)
print(result)
