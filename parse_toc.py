import json
def parse_toc(filename): 
    with open('grade_0_toc.txt', 'r') as f:
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
    return book_structure

res_0 = parse_toc('grade_0_toc.txt')
res_1 = parse_toc('grade_1_toc.txt')
res_2 = parse_toc('grade_2_toc.txt')
res_3 = parse_toc('grade_3_toc.txt')
res_4 = parse_toc('grade_4_toc.txt')

print(json.dumps(res_0, indent=4))
print(json.dumps(res_1, indent=4))
print(json.dumps(res_2, indent=4))
print(json.dumps(res_3, indent=4))
print(json.dumps(res_4, indent=4))
