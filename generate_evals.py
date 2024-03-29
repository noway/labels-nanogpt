import json
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv('KEY'))

FEW_SHOT_COUNT = 'fourteen'

def generate_1_eval(filepath):
    # Generate 3 examples
    examples = client.completions.create(model="text-davinci-003",
        prompt=f"Generate ${FEW_SHOT_COUNT} simple math problems along with their answers suitable Grade 4 K-12 students.",
        temperature=0.5,
        max_tokens=800).choices[0].text.strip()
    print (examples)
    # Format the examples into the prompt for the next generation
    prompt = f'''Generate a simple math problem along with its answer suitable Grade 4 K-12 students.

    {examples}

    Problem: '''

    problems_dict = {}

    # Generate 1 new math problems
    for i in range(1):
        response = client.completions.create(model="text-davinci-003",
            prompt=prompt,
            temperature=0.5,
            max_tokens=200)
            
        print (f'\n15. {response.choices[0].text.strip()}')
        print ('-----------------------------------')

        # Splits into problem and answer, and stores in dictionary
        problem_answer = response.choices[0].text.strip().split('\n')
        problems_dict[f'Problem_{i + 1}'] = {
            'Problem': problem_answer[0],
            'Answer': problem_answer[1] if len(problem_answer) > 1 else ''
        }

    # Save generated examples and problems into a JSON file
    with open(filepath, 'w') as f:
        json.dump({"examples": examples, "problems": problems_dict}, f, indent=4)


if __name__ == '__main__':
    for i in range(1, 6):
        generate_1_eval(f'eval_{i}.json')
