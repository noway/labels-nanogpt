import json


def interpret(file_name):
    with open(file_name, 'r') as f:
        data = json.load(f)
    eval_type = data['eval_type']
    all_count = data['all_count']
    ranking_dict = data['ranking_dict']
    rankin_sum = 0
    for key in ranking_dict:
        # if ranking_dict[key] == 0:
        #     print (f'ranking_dict[{key}] == 0 yay, its for {eval_type}')
        rankin_sum += ranking_dict[key]
    print(f'| {eval_type.rjust(12," ")} | {rankin_sum} |')


print('\n### no_labels')
print('| eval_type    | sum |')
print('| ------------ |-----|')
interpret('eval_results-no_labels-0shot_cot.json')
interpret('eval_results-no_labels-0shot_direct.json')
interpret('eval_results-no_labels-nshot_cot.json')
interpret('eval_results-no_labels-nshot_direct.json')

print('\n### with_labels')
print('| eval_type    | sum |')
print('| ------------ |-----|')
interpret('eval_results-with_labels-0shot_cot.json')
interpret('eval_results-with_labels-0shot_direct.json')
interpret('eval_results-with_labels-nshot_cot.json')
interpret('eval_results-with_labels-nshot_direct.json')

print('\n### label_embeddings')
print('| eval_type    | sum |')
print('| ------------ |-----|')
interpret('eval_results-label_embeddings-0shot_cot.json')
interpret('eval_results-label_embeddings-0shot_direct.json')
interpret('eval_results-label_embeddings-nshot_cot.json')
interpret('eval_results-label_embeddings-nshot_direct.json')
