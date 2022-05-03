import os
import subprocess


CMD_TMPLATE = "CUDA_VISIBLE_DEVICES={0} python train_pos.py " \
              "--model_path {1} " \
              "--eval_method {2} " \
              "--tag_type {3} " \
              "--word_embedding {4} " \
              "--num_epochs {5} " \
              "--batch_size {6} " \
              "--num_layers {7} " \
              "--hidden_size {8} " \
              "--dim {9} " \
              "--learning_rate {10} " \
              "--corpus {11} " \
              "--rank {12} " \
              "--std {13} " \
              "--use_tri " \
              "--use_crf " \
              "--use_rnn &" \
              #"--char_embedding" \
              #"--bert_embedding" \
              #"--elmo_embedding" \

corpus = 'UD_RUSSIAN'  # language UD_ENGLISH, UD_RUSSIAN, UD_FRENCH, UD_GERMAN, UD_ITALIAN, UD_SPANISH, UD_INDONESIAN, UD_CROATIAN
model_path = 'resources/taggers/tri/ru'  # language glove, ru, fr, de, it, es, id, hr
eval_method = 'MICRO_ACCURACY'
tag_type = 'upos'
word_embedding = 'ru'
char_embedding = False
bert_embedding = False
elmo_embedding = False
use_tri = True
use_crf = True
use_rnn = True
num_epochs = 150
batch_size = 32
num_layers = 1
hidden_size = 256
dimensions = 20
learning_rates = 0.1
rank = 396
std = 0.1545


def run_command(gpu, model_path,
                eval_method, tag_type,
                word_embedding, num_epochs,
                batch_size, num_layers,
                hidden_size, dim,
                learning_rate, corpus,
                rank, std,
                ex):
    emb = 'w'
    emb = emb + 'c' if char_embedding else emb
    emb = emb + 'b' if bert_embedding else emb
    emb = emb + 'e' if elmo_embedding else emb

    model_path = os.path.join(model_path, '_'.join([emb, str(learning_rate), str(ex)]))
    cmd = CMD_TMPLATE.format(gpu, model_path,
                             eval_method, tag_type,
                             word_embedding, num_epochs,
                             batch_size, num_layers,
                             hidden_size, dim,
                             learning_rate, corpus,
                             rank, std)

    filepath = os.path.join(model_path)
    if not (os.path.exists(filepath) and os.path.isdir(filepath)):
        os.makedirs(filepath, exist_ok=True)
    file = open(os.path.join(filepath, 'stdout.log'), 'w')
    print(cmd)

    subprocess.call(cmd, shell=True, stdout=file)


if __name__ == '__main__':

    for e in range(1):
        gpu = e + 2
        learning_rate = learning_rates
        dim = dimensions

        run_command(gpu, model_path,
                    eval_method, tag_type,
                    word_embedding, num_epochs,
                    batch_size, num_layers,
                    hidden_size, dim,
                    learning_rate, corpus,
                    rank, std,
                    e+1)
