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


def get_grade_books_content(level_codename, book_type):
    with open(f'{level_codename}_toc.json', 'r') as file:
        json_data = file.read()

    data = json.loads(json_data)
    grade = level_codename_to_level(level_codename)
    concatenated_contents = ''
    for chapter, sections in data.items():
        for section in sections:
            file_name = ''
            if book_type == 'textbook':
                file_name = f'{base_path}/{grade}_{chapter}_{section}.txt'
            elif book_type == 'workbook':
                file_name = f'{base_path}/workbook_{grade}_{chapter}_{section}.txt'
            elif book_type == 'practicebook':
                file_name = f'{base_path}/practicebook_{grade}_{chapter}_{section}.txt'
            if os.path.exists(file_name):
                with open(file_name, 'r') as file:
                    modified_content = f'\<\|document\|\>{grade} {book_type}. Chapter: {chapter}. Section: {section}.\n{file.read()}\n'
                    concatenated_contents += modified_content
    return concatenated_contents


def get_grade_full_content(level_codename):
    result = ''
    result += get_grade_books_content(level_codename, 'textbook')
    result += get_grade_books_content(level_codename, 'workbook')
    result += get_grade_books_content(level_codename, 'practicebook')
    return result


def get_all_grades_full_content():
    result = ''
    for level_codename in ['grade_0', 'grade_1', 'grade_2', 'grade_3', 'grade_4']:
        result += get_grade_full_content(level_codename)
    return result


with open('./trainingdata.txt', 'w') as file:
    file.write(get_all_grades_full_content())
