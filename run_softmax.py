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
              "--use_rnn " \
              # "--char_embedding " \
            # "--elmo_embedding" \


"""
Corpus Option:
    POS: UD_ENGLISH    UD_RUSSIAN  UD_FRENCH    UD_GERMAN  UD_ITALIAN   UD_SPANISH    UD_INDONESIAN  UD_CROATIAN
    NER: conll_03_dutch    conll_03_english    conll_03_german    conll_03_spanish 

Embeddings Options: 
    BERT: bert-base-multilingual-cased  bert-base-german-cased bert-base-uncased bert-base-cased
    Word for NER: conll_dutch conll_german conll_english conll_spanish
"""
# Change if different model or embeddings
corpus = 'conll_03_english'  # language
word_embedding = None#'conll_spanish'  # language
char_embedding = False
bert_embedding = 'bert-base-multilingual-cased'
# Change if different task
eval_method = 'f1' # acc f1
tag_type = 'ner' # ner upos
# TODO: Do not change below
model_path = tag_type + '/softmax/' + corpus  # language glove, ru, fr, de, it, es, id, hr
use_rnn = True
num_epochs = 150
batch_size = 32
num_layers = 1
hidden_size = 256
learning_rate = 0.1

if char_embedding:
    CMD_TMPLATE += '--char_embedding '

CMD_TMPLATE += '&'

def run_command(gpu, model_path,
                eval_method, tag_type,
                word_embedding, bert_embedding,
                num_epochs,
                batch_size, num_layers,
                hidden_size, learning_rate,
                corpus, ex):
    emb = 'w' if word_embedding is not None else''
    emb = emb + 'c' if char_embedding else emb
    emb = emb + 'b' if bert_embedding else emb

    model_path = os.path.join(model_path, '_'.join([emb, str(learning_rate), str(ex)]))
    cmd = CMD_TMPLATE.format(gpu, model_path,
                             eval_method, tag_type,
                             word_embedding, bert_embedding,
                             num_epochs,
                             batch_size, num_layers,
                             hidden_size, learning_rate,
                             corpus)

    filepath = os.path.join(model_path)
    if not (os.path.exists(filepath) and os.path.isdir(filepath)):
        os.makedirs(filepath, exist_ok=True)
    file = open(os.path.join(filepath, 'stdoutdev.log'), 'w')
    print(cmd)
    subprocess.call(cmd, shell=True, stdout=file)


if __name__ == '__main__':

    for e in range(1):
        gpu = (e + 0) % 4 + 0
        run_command(gpu, model_path,
                    eval_method, tag_type,
                    word_embedding, bert_embedding,
                    num_epochs,
                    batch_size, num_layers,
                    hidden_size, learning_rate,
                    corpus, e + 12)
