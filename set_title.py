import os

def modify_first_line(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as file:
                lines = file.readlines()
            
            if lines:
                lines[0] = lines[0].replace('*', '')

                if not lines[0].startswith('#'):
                    lines[0] = '# ' + lines[0]
                
                with open(filepath, 'w') as file:
                    file.writelines(lines)

directory_path = '.'
modify_first_line(directory_path)
