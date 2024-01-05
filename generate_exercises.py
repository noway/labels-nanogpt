def subtraction_exercise(n, m, drop_answer=False):
    exercise = "**Exercise**: {} - {} = ?\n".format(n, m)
    
    exercise += "Let's count:\n"
    count_values = ', '.join([str(i+1) for i in range(n)])
    exercise += count_values + "\n"

    exercise += "Count {} less:\n".format(m)
    reduced_values = ', '.join([str(n-i) for i in range(1, m+1)])
    exercise += reduced_values + "\n"

    answer = n - m
    exercise += "**Answer**: {} - {} = {}\n".format(n, m, answer if not drop_answer else "")
    exercise += "\n"
    return exercise, answer

def generate_exercises(k, l):
    result = ""
    n = 3
    for i in range(n):
        excerise, _ = subtraction_exercise(5, i+1)
        result += excerise

    excerise, answer = subtraction_exercise(k, l)
    result += excerise
    return result, answer


import random
import json

def generate_and_sort_nums():
    num1 = random.randint(0, 9)
    num2 = random.randint(0, 9)
    
    num1, num2 = sorted([num1, num2])

    return num1, num2

num1, num2 = generate_and_sort_nums()
print("First number: ", num1)
print("Second number: ", num2)

exer, answer = generate_exercises(num2, num1)

with open(f'exercises{num2}_{num1}.json', 'w') as f:
    json_str = json.dumps({'exer': exer, 'answer': answer})
    f.write(json_str)

