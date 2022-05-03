from flair.data import Corpus
from flair.datasets import UD_ENGLISH, UD_RUSSIAN, UD_FRENCH, UD_GERMAN, UD_ITALIAN, UD_SPANISH, UD_INDONESIAN, UD_CROATIAN
from flair.embeddings import TokenEmbeddings, CharacterEmbeddings, WordEmbeddings, BertEmbeddings, ELMoEmbeddings, StackedEmbeddings
from typing import List
from flair.training_utils import EvaluationMetric
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
import argparse
import flair.datasets
import torch
import torchvision
import flair.nn

parser = argparse.ArgumentParser(description='Learning with ncrf or triLinear-ncrf')
parser.add_argument('--use_rnn', action='store_true', help='using rnn')
parser.add_argument('--use_crf', action='store_true', help='using crf')
parser.add_argument('--use_tri', action='store_true', help='using tri_linear')
parser.add_argument('--dim', type=int, default=None, help='dimension of tri_linear')
parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=32, help='Number of sentences in each batch')
parser.add_argument('--hidden_size', type=int, default=100, help='Number of hidden units in RNN')
parser.add_argument('--num_layers', type=int, default=1, help='Number of layers of RNN')
parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate')
parser.add_argument('--word_embedding', default='glove', help='Embedding for words')
parser.add_argument('--char_embedding', action='store_true', help='Embedding for character')
parser.add_argument('--bert_embedding', action='store_true', help='Embedding for bert')
parser.add_argument('--elmo_embedding', action='store_true', help='Embedding for elmo')
parser.add_argument('--tag_type', default='upos', help='type of tag')
parser.add_argument('--eval_method', default='MICRO_ACCURACY', help='method of evaluation')
parser.add_argument('--model_path', default='resources/taggers/example-upos-crf', help='path of saving model')
parser.add_argument('--corpus', default='UD_ENGLISH', help='type of tag')
parser.add_argument('--rank', type=int, default=None, help='trilinear matrix rank')
parser.add_argument('--std', type=float, default=None, help='trilinear matrix std')


args = parser.parse_args()
print(args)

dataset = getattr(flair.datasets, args.corpus)
tag_type = args.tag_type  # 2. what tag do we want to predict?
word_embedding = args.word_embedding
hidden_size = args.hidden_size
use_rnn = args.use_rnn
use_tri = args.use_tri
tri_dim = args.dim
use_crf = args.use_crf
lr = args.learning_rate
rank = args.rank
std = args.std
batch_size = args.batch_size
epoch = args.num_epochs
model_path = args.model_path
eval_method = getattr(EvaluationMetric, args.eval_method)
# 1. get the corpus
corpus: Corpus = dataset()
print(corpus)

# 3. make the tag dictionary from the corpus
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
print(tag_dictionary.idx2item)

# 4. initialize embeddings
Embeds = list()

tokens = corpus._get_most_common_tokens(-1, -1)
Embeds.append(WordEmbeddings(word_embedding, tokens))
if args.char_embedding:
    Embeds.append(CharacterEmbeddings())
if args.bert_embedding:
    Embeds.append(BertEmbeddings())
if args.elmo_embedding:
    Embeds.append(ELMoEmbeddings())
embedding_types: List[TokenEmbeddings] = Embeds

embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

# 5. initialize sequence tagger

tagger: SequenceTagger = SequenceTagger(hidden_size=hidden_size,
                                        embeddings=embeddings,
                                        tag_dictionary=tag_dictionary,
                                        tag_type=tag_type,
                                        use_rnn=use_rnn,
                                        use_tri=use_tri,
                                        tag_dim=tri_dim,
                                        rank=rank,
                                        std=std,
                                        use_crf=use_crf)

# 6. initialize trainer

trainer: ModelTrainer = ModelTrainer(tagger, corpus)

# 7. start training
trainer.train(model_path,
              learning_rate=lr,
              mini_batch_size=batch_size,
              max_epochs=epoch,
              embeddings_storage_mode="gpu",
              num_workers=0)



