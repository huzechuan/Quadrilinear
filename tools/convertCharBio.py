import os
import sys
import re
from typing import List

def word2char(file):
    path_to_conll_file = os.path.join(root, file)
    convert_path = path_to_conll_file + '_bio'
    fout = open(convert_path, 'w', encoding='utf-8')
    tag_dic = {'pos': 4}
    tag_type = tag_dic['pos']
    total_sentence_count = 0
    with open(str(path_to_conll_file), encoding="utf-8") as file:

        line = file.readline()
        tokenAndTag: List[str] = list()
        while line:

            line = line.strip()
            fields: List[str] = re.split("\t+", line)
            if line == "":
                total_sentence_count += 1
                fout.write('\n')
            elif line.startswith("#"):
                line = file.readline()
                continue
            elif "." in fields[0]:
                line = file.readline()
                continue
            elif "-" in fields[0]:
                line = file.readline()
                continue
            else:
                # print(fields)
                token = fields[1]
                tag = (str(fields[tag_type]))
                length = len(token)
                for i in range(length):
                    if fields[3] in 'PUNCT':
                        fout.write(token[i] + ' ' + 'O' + '\n')
                        continue
                    if i==0:
                        fout.write(token[i] + ' ' + 'B-' + tag + '\n')
                    else:
                        fout.write(token[i] + ' ' + 'I-' + tag + '\n')

            line = file.readline()

    fout.close()
    print(f'{convert_path}-total_sents: {total_sentence_count}')
        # if len(token) > 0:
        #     self.total_sentence_count += 1
        #     self.sentences.append(sentence)
        # if len(tag) > 0:
        #     self.tags.append(tag)

if __name__=='__main__':
    root = './'
    datasets = os.listdir(root)

    for dataset in datasets:
        if '.py' in dataset:
            continue
        word2char(dataset)
