import os
import subprocess


CMD_TMPLATE = "CUDA_VISIBLE_DEVICES={0} python train.py " \
              "--model_path {1} " \
              "--eval_method {2} " \
              "--tag_type {3} " \
              "--word_embedding {4} " \
              "--bert_embedding {5} " \
              "--num_epochs {6} " \
              "--batch_size {7} " \
              "--num_layers {8} " \
              "--hidden_size {9} " \
              "--dim {10} " \
              "--learning_rate {11} " \
              "--corpus {12} " \
              "--rank {13} " \
              "--std {14} " \
              "--use_which {15} " \
              "--use_crf " \
              "--use_rnn " \
              #"--char_embedding" \
              #"--elmo_embedding" \

"""
Corpus Option:
    POS:  UD_DUTCH UD_HINDI UD_JAPANESE UD_CHINESE  UD_ENGLISH UD_ITALIAN UD_GERMAN UD_INDONESIAN   
    NER: conll_03_dutch  conll_03_german  conll_03_english    conll_03_spanish 
    chunk: conll_03_vietnamese
Model Options: Trilinear Bilinear ThreeBilinear Qualinear Pentalinear ConcatScore

Embeddings Options: 
    BERT: bert-base-multilingual-cased  bert-base-german-cased  bert-base-cased bert-base-chinese
    Word for NER: conll_dutch conll_german conll_english conll_spanish conll_vietnamese
"""
# Change if different model or embeddings
corpus = 'conll_03_dutch'  # patience
word_embedding = 'conll_dutch'
char_embedding = False
bert_embedding = None#'bert-base-multilingual-cased'
use_which = 'ConcatScore-Trilinear'
# Change if different task
eval_method = 'f1' # acc f1
tag_type = 'ner' # ner upos
# TODO: Do not change below
model_path = tag_type + '_tune/' + use_which + '/' + corpus # language glove, ru, fr, de, it, es, id, hr, nl
use_crf = True
use_rnn = True
num_epochs = 300
batch_size = 32
num_layers = 1
hidden_size = 256
dimensions = [200, 100, 50, 20]
learning_rates = 0.1
cat_dimesions = [384, 256, 128, 64]
std = 0.1545

if char_embedding:
    CMD_TMPLATE += '--char_embedding '

# CMD_TMPLATE += '\n'


def run_command(gpu, model_path,
                eval_method, tag_type,
                word_embedding, bert_embedding,
                num_epochs,
                batch_size, num_layers,
                hidden_size, dim,
                learning_rate, corpus,
                ranks, std,
                use_which):
    emb = 'w' if word_embedding is not None else''
    emb = emb + 'c' if char_embedding else emb
    emb = emb + 'b' if bert_embedding is not None else emb

    model_path = os.path.join(model_path, '_'.join([emb, str(learning_rate), str(dim)]))
    cmds = ''
    filepath = os.path.join(model_path)
    if not (os.path.exists(filepath) and os.path.isdir(filepath)):
        os.makedirs(filepath, exist_ok=True)
    file = open(os.path.join(filepath, 'stdout.log'), 'w')
    length = len(ranks)
    for lens in range(length):
        cmd = CMD_TMPLATE.format(gpu, model_path,
                                 eval_method, tag_type,
                                 word_embedding, bert_embedding,
                                 num_epochs,
                                 batch_size, num_layers,
                                 hidden_size, dim,
                                 learning_rate, corpus,
                                 ranks[lens], std, use_which)
        # if lens == 0:
        #     cmds += (cmd + '\n') * 3
        #     continue
        # if lens == 0 and rank == 64:
        #     continue
        rounds = 3
        if lens + 1 < length:
            cmds += (cmd + '\n') * rounds #+ cmd + '\n' + cmd + '\n' + cmd + '\n' + cmd + '\n'
        else:
            cmds += (cmd + '\n') * (rounds - 1) + cmd #+ '\n' + cmd + '\n' + cmd + '\n' + cmd
        print(f'{cmd} * {rounds}')
    # print(cmds)
    sh_params = f'{tag_type}_{use_which}_{corpus}_{emb}_{learning_rate}_{dim}_{gpu}'
    sh_file = open(f'./sh_file/{sh_params}_temp.sh', 'w')
    sh_file.write(cmds)
    subprocess.call(f'bash ./sh_file/{sh_params}_temp.sh &', shell=True, stdout=file)
    # file.close()


if __name__ == '__main__':
    a = [0,1, 3]
    for e in range(1):
        gpu = (e + 0) % 4 + 0
        # gpu = a[e]
        e += 3

        learning_rate = learning_rates
        dim = dimensions[e]
        rank = cat_dimesions[:]
        run_command(gpu, model_path,
                    eval_method, tag_type,
                    word_embedding, bert_embedding,
                    num_epochs,
                    batch_size, num_layers,
                    hidden_size, dim,
                    learning_rate, corpus,
                    rank, std,
                    use_which)
