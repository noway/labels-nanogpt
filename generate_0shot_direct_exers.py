import random
from ruamel.yaml import YAML

def subtraction_exercise(n, m, drop_answer=False):
    # exercise = "**Exercise**: {} - {} = ?\n".format(n, m)
    
    # exercise += "Let's count:\n"
    # count_values = ', '.join([str(i+1) for i in range(n)])
    # exercise += count_values + "\n"

    # exercise += "Count {} less:\n".format(m)
    # reduced_values = ', '.join([str(n-i) for i in range(1, m+1)])
    # exercise += reduced_values + "\n"

    exercise = ""
    answer = n - m
    exercise += "**Answer**: {} - {} = {}".format(n, m, f'{answer}\n\n' if not drop_answer else "")
    return exercise, answer

def generate_exercises(k, l):
    result = ""
    # n = 3
    # for i in range(n):
    #     excerise, _ = subtraction_exercise(5, i+1)
    #     result += excerise

    excerise, answer = subtraction_exercise(k, l, drop_answer=True)
    result += excerise
    return result, answer


yaml = YAML()
yaml.default_style = '|'

def generate_eval():
    for num1 in range(10):
        for num2 in range(10):
            if num1 < num2:
                continue
            exer, answer = generate_exercises(num1, num2)

            with open(f'exercises{num1}_{num2}.yml', 'w') as f:
                dict_exer = {'exercises': exer, 'answer': answer, eval_type: '0shot_direct'}
                yaml_str = yaml.dump(dict_exer, f)


if __name__ == "__main__":
    generate_eval()