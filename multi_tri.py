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
    POS: UD_ENGLISH    UD_RUSSIAN  UD_FRENCH    UD_GERMAN  UD_ITALIAN   UD_SPANISH    UD_INDONESIAN  UD_CROATIAN
    NER: conll_03_dutch  conll_03_german  conll_03_english    conll_03_spanish 

Model Options: Trilinear Bilinear ThreeBilinear Qualinear

Embeddings Options: 
    BERT: bert-base-multilingual-cased  bert-base-german-cased bert-base-uncased bert-base-cased
    Word for NER: conll_dutch conll_german conll_english conll_spanish
"""
# Change if different model or embeddings
corpus = 'UD_INDONESIAN'  # patience
word_embedding = 'id'
char_embedding = True
bert_embedding = None
use_which = 'Qualinear'
# Change if different task
eval_method = 'acc' # acc f1
tag_type = 'upos' # ner upos
# TODO: Do not change below
model_path = tag_type + '/' + use_which + '/' + corpus # language glove, ru, fr, de, it, es, id, hr, nl
use_crf = True
use_rnn = True
num_epochs = 300
batch_size = 32
num_layers = 1
hidden_size = 256
dimensions = [20, 50, 150, 250, 400]
learning_rates = 0.1
ranks = [32, 64, 128, 256, 384, 512, 640, 800]
std = 0.1545

if char_embedding:
    CMD_TMPLATE += '--char_embedding '

CMD_TMPLATE += '&'


def run_command(gpu, model_path,
                eval_method, tag_type,
                word_embedding, bert_embedding,
                num_epochs,
                batch_size, num_layers,
                hidden_size, dim,
                learning_rate, corpus,
                rank, std,
                use_which):
    emb = 'w' if word_embedding is not None else''
    emb = emb + 'c' if char_embedding else emb
    emb = emb + 'b' if bert_embedding is not None else emb

    model_path = os.path.join(model_path, '_'.join([emb, str(learning_rate), str(rank)]))
    cmds = ''
    filepath = os.path.join(model_path)
    if not (os.path.exists(filepath) and os.path.isdir(filepath)):
        os.makedirs(filepath, exist_ok=True)
    file = open(os.path.join(filepath, 'stdout.log'), 'w')
    length = len(dim)
    for lens in range(length):
        cmd = CMD_TMPLATE.format(gpu, model_path,
                                 eval_method, tag_type,
                                 word_embedding, bert_embedding,
                                 num_epochs,
                                 batch_size, num_layers,
                                 hidden_size, dim[lens],
                                 learning_rate, corpus,
                                 rank, std, use_which)
        if lens + 1 < length:
            cmds += cmd + '& ' + cmd + '& ' + cmd + '& '
        else:
            cmds += cmd + '& ' + cmd + '& ' + cmd
    print(cmds)

    subprocess.call(cmds, shell=True, stdout=file)
    # file.close()


if __name__ == '__main__':

    for e in range(1):
        gpu = (e + 3) % 4 + 0

        learning_rate = learning_rates
        dim = dimensions
        rank = ranks[7]
        run_command(gpu, model_path,
                    eval_method, tag_type,
                    word_embedding, bert_embedding,
                    num_epochs,
                    batch_size, num_layers,
                    hidden_size, dim,
                    learning_rate, corpus,
                    rank, std,
                    use_which)
