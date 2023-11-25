import json
section_count = 0
def parse_toc(filename): 
    global section_count
    with open(filename, 'r') as f:
        text = f.read()

    lines = text.strip().split('\n')

    book_structure = {}

    current_chapter_number = None
    current_chapter_title = None

    for line in lines:
        if line.strip():
            if '.' in line and ' ' in line:
                parts = line.split(' ', 1)
                chapter_number = parts[0].split('.')[0]

                if chapter_number != current_chapter_number:
                    current_chapter_number = chapter_number
                    current_chapter_title = parts[1].strip()
                    book_structure[current_chapter_title] = []
                else:
                    section_title = parts[1].strip()
                    book_structure[current_chapter_title].append(section_title)
                    section_count += 1
    return book_structure

res_0 = parse_toc('grade_0_toc.txt')
res_1 = parse_toc('grade_1_toc.txt')
res_2 = parse_toc('grade_2_toc.txt')
res_3 = parse_toc('grade_3_toc.txt')
res_4 = parse_toc('grade_4_toc.txt')

print(section_count)

with open('grade_0_toc.json', 'w') as f:
    json.dump(res_0, f, indent=4)
with open('grade_1_toc.json', 'w') as f:
    json.dump(res_1, f, indent=4)
with open('grade_2_toc.json', 'w') as f:
    json.dump(res_2, f, indent=4)
with open('grade_3_toc.json', 'w') as f:
    json.dump(res_3, f, indent=4)
with open('grade_4_toc.json', 'w') as f:
    json.dump(res_4, f, indent=4)
