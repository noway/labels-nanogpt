import json
from openai import OpenAI
import sys
import os
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(api_key=os.getenv("KEY"))

def ask_gpt4_chat(question):
    try:
        response = client.chat.completions.create(model="gpt-4-1106-preview",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question}
        ])
        return response.choices[0].message.content
    except Exception as e:
        return str(e)

def generate_section_of_a_chapter(level, chapter, section):
    question = f'You\'re writing a math textbook for {level} level K-12. Write the "{section}" section of the "{chapter}" chapter. Make sure the section is thorough and complete, including example excercises and answeres.'
    answer = ask_gpt4_chat(question)
    return answer

def generate_and_save_section_of_a_chapter(level, chapter, section):
    text = generate_section_of_a_chapter(level, chapter, section)
    print(text)
    filename = f'{level}_{chapter}_{section}.txt'
    with open(filename, 'w') as f:
        f.write(text)

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

level_codename = sys.argv[1]
with open(f'{level_codename}_toc.json') as f:
    data = json.load(f)
    level = level_codename_to_level(level_codename)
    for chapter in data:
        for section in data[chapter]:
            generate_and_save_section_of_a_chapter(level, chapter, section)
