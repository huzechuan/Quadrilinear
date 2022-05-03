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
              "--learning_rate {10} " \
              "--corpus {11} " \
              "--percent {12} " \
              "--L2 {13} " \
              "--use_rnn " \
              #"--char_embedding" \
              #"--elmo_embedding" \

"""
Corpus Option:
    POS: UD_ENGLISH    UD_RUSSIAN  UD_FRENCH    UD_GERMAN  UD_ITALIAN   UD_SPANISH    UD_INDONESIAN  UD_CROATIAN
    NER: conll_03_dutch  conll_03_german  conll_03_english    conll_03_spanish 

Embeddings Options: 
    BERT: bert-base-multilingual-cased  bert-base-german-cased bert-base-uncased bert-base-cased
    Word for NER: conll_dutch conll_german conll_english conll_spanish
"""
# Change if different model or embeddings
corpus = ['conll_03_english', 'conll_03_german', 'conll_03_vietnamese']  # language
word_embeddings = ['conll_english', 'conll_german', 'conll_vietnamese'] #'conll_spanish'  # language
char_embedding = False#True
bert_embeddings = ['bert-base-cased', 'bert-base-german-cased', 'bert-base-multilingual-cased']
# Change if different task
eval_method = 'f1' # acc f1
tag_type = 'chunk' # ner upos
# TODO: Do not change below
# model_path = tag_type + '/' + 'softmax' + '/' + corpus # language glove, ru, fr, de, it, es, id, hr, nl
use_rnn = True
num_epochs = 300
batch_size = 32
num_layers = 1
hidden_size = 256
learning_rate = 0.1
L2 = 1e-8

if char_embedding:
    CMD_TMPLATE += '--char_embedding '

# CMD_TMPLATE += '&'


def run_command(gpu, model_path,
                eval_method, tag_type,
                word_embedding, bert_embedding,
                num_epochs,
                batch_size, num_layers,
                hidden_size,
                learning_rate, corpu, percent, L2):
    emb = 'w' if word_embedding is not None else''
    emb = emb + 'c' if char_embedding else emb
    emb = emb + 'b' if bert_embedding is not None else emb
    if percent is not 1:
        emb += str(percent)
    model_path = os.path.join(model_path, '_'.join([str(f'L2={L2}'), emb, str(learning_rate)]))
    cmds = ''
    filepath = os.path.join(model_path)
    if not (os.path.exists(filepath) and os.path.isdir(filepath)):
        os.makedirs(filepath, exist_ok=True)
    file = open(os.path.join(filepath, 'stdout.log'), 'w')

    rounds = 5
    for lens in range(rounds):
        cmd = CMD_TMPLATE.format(gpu, model_path,
                                 eval_method, tag_type,
                                 word_embedding, bert_embedding,
                                 num_epochs,
                                 batch_size, num_layers,
                                 hidden_size,
                                 learning_rate, corpu, percent, L2)
        if lens + 1 < rounds:
            cmds += cmd + '\n'
        else:
            cmds += cmd
            print(f'{cmd} * {rounds}')

    sh_params = f'{tag_type}_softmax_{corpu}_{emb}_L2={L2}_{learning_rate}_{gpu}'
    sh_file = open(f'./sh_file/{sh_params}_temp.sh', 'w')
    sh_file.write(cmds)
    subprocess.call(f'bash ./sh_file/{sh_params}_temp.sh &', shell=True, stdout=file)
    # file.close()


if __name__ == '__main__':

    for e in range(1):
        gpu = (e + 3) % 4 + 0
        e += 2

        word_embedding = word_embeddings[e]
        # word_embedding = None
        # bert_embedding = bert_embeddings[e]
        bert_embedding = None
        corpu = corpus[e]
        model_path = tag_type + '/' + 'softmax' + '/' + corpu
        run_command(gpu, model_path,
                    eval_method, tag_type,
                    word_embedding, bert_embedding,
                    num_epochs,
                    batch_size, num_layers,
                    hidden_size,
                    learning_rate, corpu, percent=1, L2=L2)
