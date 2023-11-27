import os

def remove_dashes(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    while lines and lines[0].strip() == "---":
        lines.pop(0)
    while lines and lines[-1].strip() == "---":
        lines.pop()

    with open(file_path, 'w') as file:
        file.writelines(lines)

def process_directory(directory_path):
    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory_path, filename)
            remove_dashes(file_path)
            print(f"Processed {filename}")

directory_path = '.'
process_directory(directory_path)
