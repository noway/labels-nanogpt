import os
import json

base_path = './dataset_md2md' 

def level_codename_to_level(level_codename):
    if level_codename == 'grade_0':
        return 'Kindergarten'
    elif level_codename == 'grade_1':
        return '1st Grade'
    elif level_codename == 'grade_2':
        return '2nd Grade'
    elif level_codename == 'grade_3':
        return '3rd Grade'
    elif level_codename == 'grade_4':
        return '4th Grade'

def get_grade_full_content(level_codename):
    with open(f'{level_codename}_toc.json', 'r') as file:
        json_data = file.read()

    data = json.loads(json_data)
    grade = level_codename_to_level(level_codename)
    concatenated_contents = ""
    for chapter, sections in data.items():
        for section in sections:
            file_name = f"{base_path}/{grade}_{chapter}_{section}.txt"
            if os.path.exists(file_name):
                with open(file_name, 'r') as file:
                    modified_content = f"\<\|document\|\>{grade} textbook. Chapter: {chapter}. Section: {section}.\n{file.read()}\n"
                    concatenated_contents += modified_content
    return concatenated_contents

with open('grade_0_concatenated.txt', 'w') as file:
    file.write(get_grade_full_content('grade_0'))