import json
from openai import OpenAI
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
    question = f'You\'re writing a math textbook for {level} level K-12. Write the "{section}" section of the "{chapter}" chapter. Make sure the section is thorough and complete, including example excercises and answeres. Don\'t use emojis.'
    answer = ask_gpt4_chat(question)
    return answer

def generate_and_save_section_of_a_chapter(level, chapter, section):
    text = generate_section_of_a_chapter(level, chapter, section)
    print(text)
    filename = f'{level}_{chapter}_{section}.txt'
    with open(filename, 'w') as f:
        f.write(text)

generate_and_save_section_of_a_chapter('Kindergarten', 'Counting from 1 to 10', 'Introduction to Numbers')