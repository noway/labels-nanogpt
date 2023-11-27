import os
import json

with open('grade_0_toc.json', 'r') as file:
    json_data = file.read()

data = json.loads(json_data)
base_path = './dataset_md2md' 
concatenated_contents = ""

for chapter, sections in data.items():
    for section in sections:
        grade = "Kindergarten"
        file_name = f"{base_path}/{grade}_{chapter}_{section}.txt"
        if os.path.exists(file_name):
            with open(file_name, 'r') as file:
                modified_content = f"\<\|document\|\>{grade} textbook. Chapter: {chapter}. Section: {section}.\n{file.read()}\n"
                concatenated_contents += modified_content

with open('grade_0_concatenated.txt', 'w') as file:
    file.write(concatenated_contents)