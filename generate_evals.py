import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv('KEY'))

problems_dict = {}

for i in range(5):
    response = client.completions.create(
        model="text-davinci-003",
        prompt="Generate a simple math problem along with its answer suitable for Kindergarten to Grade 4 students.\n\nProblem: ",
        temperature=0.5,
        max_tokens=50)

    problem_answer = response.choices[0].text.strip().split('\n')
    # filter empty strings from problem_answer list
    problem_answer = list(filter(None, problem_answer))
    print(problem_answer)
    problems_dict[f'Problem_{i + 1}'] = {
        'Problem': problem_answer[0],
        'Answer': problem_answer[1] if len(problem_answer) > 1 else ''
    }

# Save generated problems and answers into a JSON file
with open('problems.json', 'w') as f:
    json.dump(problems_dict, f)