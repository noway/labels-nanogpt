import os


def modify_files_in_directory(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            filepath = os.path.join(directory, filename)

            with open(filepath, 'r') as file:
                lines = file.readlines()

            if lines:
                if lines[0].startswith('###'):
                    lines[0] = lines[0].replace('###', '#', 1)

                    with open(filepath, 'w') as file:
                        file.writelines(lines)

                if lines[0].startswith('##'):
                    lines[0] = lines[0].replace('##', '#', 1)

                    with open(filepath, 'w') as file:
                        file.writelines(lines)


directory_path = '.'
modify_files_in_directory(directory_path)
