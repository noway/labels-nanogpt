import json

with open('labels.json') as f:
    labels = json.load(f)

with open('labels_non_vectorized.json') as f:
    labels_non_vectorized = json.load(f)

result = {
    label_non_vectorized: label
    for label, label_non_vectorized in zip(labels, labels_non_vectorized)
}
print(result)

with open('../labels_map.json', 'w') as f:
    json.dump(result, f, indent=4)
