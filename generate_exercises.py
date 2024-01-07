import random
# import json
import yaml

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

def generate_and_sort_nums():
    num1 = random.randint(0, 9)
    num2 = random.randint(0, 9)
    
    num1, num2 = sorted([num1, num2])

    return num1, num2

def generate_eval():
    for num1 in range(10):
        for num2 in range(10):
            if num1 < num2:
                continue
            exer, answer = generate_exercises(num1, num2)

            with open(f'exercises{num1}_{num2}.yml', 'w') as f:
                yaml_str = yaml.dump({'exercises': exer, 'answer': answer})
                f.write(yaml_str)


if __name__ == "__main__":
    generate_eval()