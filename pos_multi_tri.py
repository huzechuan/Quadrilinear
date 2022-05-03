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
              "--percent {16} " \
              "--L2 {17} " \
              "--use_crf " \
              "--use_rnn " \
              #"--char_embedding" \
              #"--elmo_embedding" \

"""
Corpus Option:
    POS: UD_ENGLISH    UD_RUSSIAN  UD_FRENCH    UD_GERMAN  UD_ITALIAN   UD_SPANISH    UD_INDONESIAN  UD_CROATIAN
    NER: conll_03_dutch  conll_03_german  conll_03_english    conll_03_spanish 

Model Options: Trilinear Bilinear ThreeBilinear Qualinear FullTrilinear

Embeddings Options: 
    BERT: bert-base-multilingual-cased  bert-base-german-cased bert-base-uncased bert-base-cased
    Word for NER: conll_dutch conll_german conll_english conll_spanish
"""
# Change if different model or embeddings
# corpus = ['UD_DUTCH', 'UD_HINDI', 'UD_JAPANESE', 'UD_CHINESE']  # language
# word_embeddings = ['nl', 'hi', 'ja', 'conll_chinese'] #'conll_spanish'  # language
# bert_embeddings = ['bert-base-multilingual-cased', 'bert-base-multilingual-cased', 'bert-base-multilingual-cased', 'bert-base-chinese']

corpus = ['UD_ENGLISH', 'UD_ITALIAN', 'UD_GERMAN', 'UD_INDONESIAN']
bert_embeddings = ['bert-base-cased', 'bert-base-multilingual-cased', 'bert-base-german-cased', 'bert-base-multilingual-cased']
word_embeddings = ['glove', 'it', 'de', 'id']

char_embedding = False
use_which = 'Qualinear'
# Change if different task
eval_method = 'acc' # acc f1
tag_type = 'pos' # ner upos pos
# TODO: Do not change below
# model_path = tag_type + '/' + use_which + '/' + corpus # language glove, ru, fr, de, it, es, id, hr, nl
use_crf = True
use_rnn = True
num_epochs = 300
batch_size = 32
num_layers = 3
hidden_size = 256
dimensions = [20, 50, 200, 250, 400]
learning_rates = 0.1
ranks = [64, 128, 256, 384, 512, 640, 800]
std = 0.1545
L2 = 1e-8

if char_embedding:
    CMD_TMPLATE += '--char_embedding '

# CMD_TMPLATE += '&'


def run_command(gpu, model_path,
                eval_method, tag_type,
                word_embedding, bert_embedding,
                num_epochs,
                batch_size, num_layers,
                hidden_size, dim,
                learning_rate, corpu,
                rank, std,
                use_which, percent, L2):
    emb = 'w' if word_embedding is not None else''
    emb = emb + 'c' if char_embedding else emb
    emb = emb + 'b' if bert_embedding is not None else emb
    if percent is not 1:
        emb += str(percent)
    model_path = os.path.join(model_path, '_'.join([str(f'L2={L2}'), emb, str(learning_rate), str(rank), str(dim)]))
    cmds = ''
    filepath = os.path.join(model_path)
    if not (os.path.exists(filepath) and os.path.isdir(filepath)):
        os.makedirs(filepath, exist_ok=True)
    file = open(os.path.join(filepath, 'stdout.log'), 'w')

    rounds = 3
    for lens in range(rounds):
        cmd = CMD_TMPLATE.format(gpu, model_path,
                                 eval_method, tag_type,
                                 word_embedding, bert_embedding,
                                 num_epochs,
                                 batch_size, num_layers,
                                 hidden_size, dim,
                                 learning_rate, corpu,
                                 rank, std, use_which, percent, L2)
        if lens + 1 < rounds:
            cmds += cmd + '\n'
        else:
            cmds += cmd
            print(f'{cmd} * {rounds}')

    sh_params = f'{tag_type}_{use_which}_{corpu}_{emb}_L2={L2}_{learning_rate}_{rank}_{dim}_{gpu}'
    sh_file = open(f'./sh_file/{sh_params}_temp.sh', 'w')
    sh_file.write(cmds)
    subprocess.call(f'bash ./sh_file/{sh_params}_temp.sh &', shell=True, stdout=file)
    # file.close()


if __name__ == '__main__':

    for e in range(2):
        gpu = (e + 2) % 4 + 0
        e += 2

        learning_rate = learning_rates
        dim = dimensions[0]
        rank = ranks[e]

        word_embedding = word_embeddings[2]
        # word_embedding = None
        # bert_embedding = bert_embeddings[e]
        bert_embedding = None

        corpu = corpus[2]
        model_path = tag_type + '/' + use_which + '/' + corpu

        run_command(gpu, model_path,
                    eval_method, tag_type,
                    word_embedding, bert_embedding,
                    num_epochs,
                    batch_size, num_layers,
                    hidden_size, dim,
                    learning_rate, corpu,
                    rank, std,
                    use_which, percent=1, L2=L2)
