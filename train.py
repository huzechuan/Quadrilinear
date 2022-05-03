from __future__ import print_function
import time
import torch
import torch.nn as nn
from models.crf import *
from models.lstm_model import *
from models.evaluator import *
import argparse
import os
import sys
import logging
import itertools
import data_process.datasets as datasets
from data_process.data import (
    Corpus
)
from data_process.datasets import UD_ENGLISH, UD_RUSSIAN
from Embeddings.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, CharacterEmbeddings, BertEmbeddings, XLMRobertaEmbeddings

from typing import List
from models.utils import log_info
from data_process import datasets as DT
from torch.optim.lr_scheduler import ReduceLROnPlateau

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


# def log_info(info):
#     now_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
#     print(f"\033[0;31m {now_time} ")
#     print(info)
def repack(batch):
    t = []
    m = []
    for b in batch:
        t.append(b.tag)
        m.append(b.mask)
    tags = torch.cat(t, 0)
    masks = torch.cat(m, 0)
    return tags.transpose(0, 1).unsqueeze(2).cuda(), masks.transpose(0, 1).cuda()

#
# def log_line():
#     log_info("-" * 100)

def collate_fn(batch):
    return batch

def train(use_crf = True,
          use_which = None,
          tag_dim = None,
          max_epoch = 150,
          mini_batch_size = 32,
          num_layers=1,
          hidden_size = 256,
          learning_rate = 0.1,
          word_embedding = 'glove',
          if_cased = False,
          char_embedding = None,
          bert_embedding = None,
          tag_type='upos',
          eval_method = 'acc',
          path = './resources/example-upos-crf',
          corpu = None,
          rank = None,
          std = None,
          L2=1e-8,
          percentage=1,
          encoder='lstm',
          neighbor='prev',
          normalize=False,
          device='cpu',
          save_model=True,
          scheme='BIOES'
          ):
    save_model = False
    tri_linear_parameter = {'use_which': use_which,
                            'tag_dim': tag_dim,
                            'rank': rank,
                            'std': std,
                            'neighbor': neighbor,
                            'normalize': normalize}
    patience = 10
    print(f'patience: {patience}')
    shuffle = True
    if tag_type in 'upos':
        corpus: Corpus = corpu(tag_type=tag_type)
    else:
        corpus: Corpus = corpu[0](corpu[1], tag_type=tag_type)
    print(corpus)

    # 2. what tag do we want to predict?

    # 3. make the tag dictionary from the corpus
    tag_dictionary = corpus.make_tag_dictionary()
    tokens = corpus.get_train_full_tokenset(-1, -1)
    print(tag_dictionary)

    # 4. initialize embeddings
    embeddings_tmp: List[TokenEmbeddings] = []
    if (word_embedding != 'None') and (word_embedding != ' ') and (word_embedding is not None):
        word_embeddings = WordEmbeddings(word_embedding, tokens, if_cased=if_cased)
        embeddings_tmp.append(word_embeddings)
    if char_embedding:
        embeddings_tmp.append(CharacterEmbeddings(tokens[1]))
    if (bert_embedding != 'None') and (bert_embedding != ' ') and (bert_embedding != None):
        if bert_embedding.startswith('bert'):# or 'BERT' in bert_embedding or 'bert' in bert_embedding:
            embeddings_tmp.append(BertEmbeddings(bert_embedding))  # 'bert-base-multilingual-cased'  'bert-base-german-cased'
        else:
            embeddings_tmp.append(XLMRobertaEmbeddings(bert_embedding, device=device))

    embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embeddings_tmp)
    sys.stdout.flush()
    # 5. process corpus.
    dataset, dev_dataset, test_dataset = corpus.construct_data(use_crf, tag_dictionary, percentage)
    dataset_loader = [torch.utils.data.DataLoader(tup,
                                                  batch_size=mini_batch_size,
                                                  shuffle=shuffle,
                                                  drop_last=False,
                                                  collate_fn=collate_fn) for tup in dataset]
    dev_dataset_loader = [torch.utils.data.DataLoader(tup,
                                                      50,
                                                      shuffle=False,
                                                      drop_last=False,
                                                      collate_fn=collate_fn) for tup in dev_dataset]
    test_dataset_loader = [torch.utils.data.DataLoader(tup,
                                                       50,
                                                       shuffle=False,
                                                       drop_last=False,
                                                       collate_fn=collate_fn) for tup in test_dataset]

    # build model
    if encoder == 'lstm':
        ner_model = LSTM_Model(tag_dictionary, embeddings, hidden_size * 2, num_layers, 0.5,
                               use_crf=use_crf, tri_parameter=tri_linear_parameter)
    elif encoder == 'transformer':
        ner_model = TransformerEncoder(tag_dictionary, embeddings, hidden_size * 2, num_layers, 0.5,
                                       use_crf=use_crf, tri_parameter=tri_linear_parameter)
    ner_model.rand_init()

    optimizer = optim.SGD(ner_model.parameters(), lr=learning_rate, weight_decay=L2)

    # crit = CRFLoss_vb(len(tag_dictionary), tag_dictionary['<START>'], tag_dictionary['<PAD>'])

    # crit.cuda()
    ner_model.cuda()

    log_info('-' * 100)
    log_info('-' * 100)
    log_info(f'Schemes: "{scheme}"')
    log_info(f'Corpus: "{corpus}"')
    log_info('-' * 100)
    log_info(f'Embedding: "{embeddings}"')
    log_info(f'Model: "{ner_model}"')
    log_info('-' * 100)
    log_info("Parameters:")
    log_info(f' - learning_rate: "{learning_rate}"')
    log_info(f' - mini_batch_size: "{mini_batch_size}"')
    log_info(f' - patience: "{patience}"')
    log_info(f' - max_epochs: "{max_epoch}"')
    log_info(f' - shuffle: "{shuffle}"')
    log_info('-' * 100)
    log_info(f'Model training base path: "{path}"')
    log_info('-' * 100)
    # log.info(f"Device: {device}")
    log_info('-' * 100)

    best_f1 = float('-inf')
    best_test_f1 = float('inf')
    best_acc = float('-inf')
    best_test_acc = float('inf')
    track_list = list()
    epoch_list = range(0, max_epoch)
    patience_count = 0

    if use_crf:
        evaluator = eval_w(tag_dictionary, eval_method, model_path, encoder, scheme=scheme)
    else:
        evaluator = eval_softmax(tag_dictionary, eval_method, model_path, scheme=scheme)

    previous_learning_rate = learning_rate

    scheduler: ReduceLROnPlateau = ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        patience=patience,
        mode='max',
        verbose=True,
    )

    total_number_of_batches = sum([len(bucket) for bucket in dataset_loader])
    modulo = max(1, int(total_number_of_batches / 10))
    for epoch_idx, start in enumerate(epoch_list):

        epoch_loss = 0
        ner_model.train()

        seen_batches = 0

        log_info('-' * 100)

        # get new learning rate
        for group in optimizer.param_groups:
            learning_rate = group["lr"]


        # reload last best model if annealing with restarts is enabled

        # stop training if learning rate becomes too small
        if learning_rate < 0.0001:
            log_info('-' * 100)
            log_info("learning rate too small - quitting training!")
            log_info('-' * 100)
            break
        batch_time = 0
        log_info(f'learning_rate: {learning_rate:.4f}')

        for batch in itertools.chain.from_iterable(dataset_loader):
            # fea_v = embeddings.embed(feature)
            start_time = time.time()
            tg_v, mask_v = repack(batch)

            ner_model.zero_grad()
            if encoder == 'lstm':
                scores, hidden = ner_model.forward(batch)
            elif encoder == 'transformer':
                scores, hidden = ner_model.forward(batch, mask_v)

            loss = ner_model.crit(scores, tg_v, mask_v)
            loss.backward()

            nn.utils.clip_grad_norm_(ner_model.parameters(), 5.0)
            optimizer.step()
            epoch_loss += loss

            seen_batches += 1
            batch_time += time.time() - start_time
            if seen_batches % modulo == 0:
                log_info(
                    f"epoch {epoch_idx + 1} - iter {seen_batches}/{total_number_of_batches} - loss "
                    f"{epoch_loss / seen_batches:.8f} - samples/sec: {mini_batch_size * modulo / batch_time:.2f}"
                )
                batch_time = 0

        epoch_loss /= total_number_of_batches

        # eval & save check_point

        if 'f1' in eval_method:
            dev_f1, dev_pre, dev_rec, _ = evaluator.calc_score(ner_model, dev_dataset_loader, 'dev')

            if dev_f1 > best_f1:
                patience_count = 0
                best_f1 = dev_f1

                test_f1, test_pre, test_rec, _ = evaluator.calc_score(ner_model, test_dataset_loader, 'test')
                best_test_f1 = test_f1
                log_info(
                    f'loss: {epoch_loss:.4f} '
                    f'dev F1: {dev_f1:.2f}, test F1: {test_f1:.2f} '
                    f'saving.............'
                )

            else:
                patience_count += 1
                log_info(
                    f'loss: {epoch_loss:.4f} '
                    f'dev F1: {dev_f1:.2f} '
                )

        else:
            dev_acc = evaluator.calc_score(ner_model, dev_dataset_loader)

            if dev_acc > best_acc:
                test_acc = evaluator.calc_score(ner_model, test_dataset_loader)
                patience_count = 0
                best_acc = dev_acc
                best_test_acc = test_acc

                log_info(
                    f'loss: {epoch_loss:.4f} '
                    f'dev_acc = {dev_acc * 100:.2f}, test_acc = {test_acc * 100:.2f} '
                    f'saving...'
                )

                if save_model:
                    torch.save({
                        'ner_model': ner_model.state_dict(),
                        'optimizer': optimizer.state_dict()

                    }, path + "/best-model.pt")

            else:
                patience_count += 1
                log_info(
                    f'loss: {epoch_loss:.4f} '
                    f'dev_acc = {dev_acc * 100:.2f} '
                )
                # print(
                #     'loss: %.4f, train acc = %.4f, dev acc = %.4f' %
                #     (epoch_loss,
                #      train_acc,
                #      dev_acc))
                track_list.append({'loss': epoch_loss, 'dev_acc': dev_acc})

        # log_info(ner_model.lstm_time)
        # log_info(ner_model.top_time)
        dev_score = dev_f1 if 'f1' in eval_method else dev_acc
        scheduler.step(dev_score)

    # print best
    if 'f1' in eval_method:
        train_f1, train_pre, train_rec, _ = evaluator.calc_score(ner_model, dataset_loader, 'train')
        log_info(
            f'Train F1: {train_f1:.2f} '
            f'Dev F1: {best_f1:.2f} '
            f'Test F1: {best_test_f1:.2f}'
        )
        tune = True
        if tune:
            with open(model_path + '/result.log', 'a') as f:
                f.writelines(
                    '-' * 100 + '\n'
                                f'rank: {rank}, tag_dim: {tag_dim}\n'
                                f'Train_acc: {train_f1:.2f} '
                                f'Dev_acc: {best_f1:.2f} '
                                f'Test_acc: {best_test_f1:.2f}\n'
                )
    else:
        # print(checkpoint + ' dev_acc: %.4f test_acc: %.4f\n' % (best_acc, best_test_acc))
        train_acc = evaluator.calc_score(ner_model, dataset_loader)
        log_info(
            f'Train_acc: {train_acc * 100:.2f} '
            f'Dev_acc: {best_acc * 100:.2f} '
            f'Test_acc: {best_test_acc * 100:.2f}'
        )
        tune = True
        if tune:
            with open(model_path + '/result.log', 'a') as f:
                f.writelines(
                    '-' * 100 + '\n'
                                f'rank: {rank}, tag_dim: {tag_dim}\n'
                                f'Train_acc: {train_acc * 100:.2f} '
                                f'Dev_acc: {best_acc * 100:.2f} '
                                f'Test_acc: {best_test_acc * 100:.2f}\n'
                )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Learning with ncrf or triLinear-ncrf')
    parser.add_argument('--use_rnn', action='store_true', help='using rnn')
    parser.add_argument('--use_crf', action='store_true', help='using crf')
    parser.add_argument('--use_which', default=None, help='using tri_linear ?')
    parser.add_argument('--dim', type=int, default=None, help='dimension of tri_linear')
    parser.add_argument('--num_epochs', type=int, default=300, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Number of sentences in each batch')
    parser.add_argument('--hidden_size', type=int, default=256, help='Number of hidden units in RNN')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of layers of RNN')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--word_embedding', default='conll_english', help='Embedding for words')
    parser.add_argument('--if_cased', action='store_true', help='if word Embedding cased or not')
    parser.add_argument('--char_embedding', action='store_true', help='Embedding for character')
    parser.add_argument('--bert_embedding', default=None, help='Embedding for bert')
    parser.add_argument('--tag_type', default='ner', help='type of tag')
    parser.add_argument('--eval_method', default='f1', help='method of evaluation')
    parser.add_argument('--model_path', default='resources/crf/de', help='path of saving model')
    parser.add_argument('--corpus', default='conll_03_english', help='type of tag')
    parser.add_argument('--rank', type=int, default=None, help='trilinear matrix rank')
    parser.add_argument('--std', type=float, default=None, help='trilinear matrix std')
    parser.add_argument('--L2', type=float, default=1e-8, help='L2')
    parser.add_argument('--percent', type=float, default=1, help='trilinear matrix std')
    parser.add_argument('--encoder', default='lstm', help='type of encoder')
    parser.add_argument('--neighbor', default='prev', help='type of tag')
    parser.add_argument('--normalize', action='store_true', help='Embedding for character')
    parser.add_argument('--scheme', default='BIOES', help='type of tag')

    args = parser.parse_args()
    print(args)

    tag_type = args.tag_type  # 2. what tag do we want to predict?
    if_cased = args.if_cased
    if tag_type in 'upos':
        datas = getattr(datasets, args.corpus)
    else:
        datas = [datasets.ConllCorpus, args.corpus]
        if args.tag_type in 'ccg':
            if_cased = False
        else:
            if_cased = True
        
    word_embedding = args.word_embedding
    char_embedding = args.char_embedding
    bert_embedding = args.bert_embedding
    hidden_size = args.hidden_size
    num_layers = args.num_layers
    use_rnn = args.use_rnn
    use_which = args.use_which
    tri_dim = args.dim
    use_crf = args.use_crf
    lr = args.learning_rate
    rank = args.rank
    std = args.std
    batch_size = args.batch_size
    epoch = args.num_epochs
    model_path = args.model_path
    eval_method = args.eval_method
    L2 = args.L2
    percent = args.percent
    neighbor = args.neighbor
    normalize = args.normalize
    # from occupy import occumpy_mem
    #
    # occumpy_mem(3)
    # use_crf = True
    # use_tri = True
    # tri_dim = 20
    # rank = 396
    # std = 0.1543
    # gpu = 2
    # torch.cuda.set_device(gpu)
    # device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # char_embedding = True
    # use_which = 'Qualinear'
    # bert_embedding = '/home/yongjiang.jy/workspace/transformers-master/examples/token-classification/icbu-en-ner-v2.1-XMLR-base-large-model'
    # word_embedding = None# 'glove'
    # tag_type = 'office'
    # datas = [datasets.ConllCorpus, 'icbu-new']
    # percent = 0.3
    # word_embedding = 'cc.el'
    # eval_method = 'acc'
    # if_cased = True
    # use_crf = True
    # noinspection PyBroadException
    train(use_crf=use_crf,
          use_which=use_which,
          tag_dim=tri_dim,
          max_epoch=epoch,
          mini_batch_size=batch_size,
          num_layers=num_layers,
          hidden_size=hidden_size,
          learning_rate=lr,
          word_embedding=word_embedding,
          if_cased=if_cased,
          char_embedding=char_embedding,
          bert_embedding=bert_embedding,
          tag_type=tag_type,
          eval_method=eval_method,
          path=model_path,
          corpu=datas,
          rank=rank,
          std=std,
          L2=L2,
          percentage=percent,
          encoder=args.encoder,
          neighbor=neighbor,
          normalize=normalize,
          device=device,
          scheme=args.scheme)
    exit(0)
    try:
        train(use_crf=use_crf,
              use_which=use_which,
              tag_dim=tri_dim,
              max_epoch=epoch,
              mini_batch_size=batch_size,
              num_layers=num_layers,
              hidden_size=hidden_size,
              learning_rate=lr,
              word_embedding=word_embedding,
              if_cased=if_cased,
              char_embedding=char_embedding,
              bert_embedding=bert_embedding,
              tag_type=tag_type,
              eval_method=eval_method,
              path=model_path,
              corpu=datas,
              rank=rank,
              std=std,
              L2=L2,
              percentage=percent,
              encoder=args.encoder,
              neighbor=neighbor,
              normalize=normalize)
    except Exception as errorInfo:
        import time
        now_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        if not os.path.exists('./ErrorInfo'):
            os.makedirs('./ErrorInfo', exist_ok=True)
        errorLog = os.path.join('./ErrorInfo', f'{now_time}.log')

        print(f'ErrorInfo: \n{errorInfo}\n')
        with open(errorLog, 'a') as fout:
            fout.writelines(
                '-' * 100 + '\n'
                f'model_path: {model_path}, rank: {rank}, dim: {str(tri_dim)}\n'
                f'ErrorInfo: \n{errorInfo}\n'
            )

