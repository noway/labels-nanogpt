import os

def clean_text_files_v2(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            filepath = os.path.join(directory, filename)

            with open(filepath, 'r') as file:
                lines = file.readlines()

            while lines and not lines[0].strip():
                lines.pop(0)
            while lines and not lines[-1].strip():
                lines.pop()

            with open(filepath, 'w') as file:
                file.writelines(lines)

directory = "."
clean_text_files_v2(directory)
