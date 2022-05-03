"""
.. module:: evaluator
    :synopsis: evaluation method (f1 score and accuracy)
 
.. moduleauthor:: Liyuan Liu, Frank Xu
"""

import torch
import numpy as np
import itertools

import models.utils as utils
from torch.autograd import Variable
from tqdm import tqdm
import sys
from models.crf import CRFDecode_vb
from models.crf import CRFSECDecode_vb
import os
import subprocess
import re


def repack(batch):
    t = []
    m = []
    for b in batch:
        t.append(b.tag)
        m.append(b.mask)
    tags = torch.cat(t, 0)
    masks = torch.cat(m, 0)
    return tags, masks.transpose(0, 1).cuda()

class eval_batch:
    """Base class for evaluation, provide method to calculate f1 score and accuracy 

    args: 
        packer: provide method to convert target into original space [TODO: need to improve] 
        l_map: dictionary for labels    
    """

    def __init__(self, l_map, file_path=None, scheme='BIOES'):
        self.l_map = l_map
        self.r_l_map = utils.revlut(l_map)
        self.file_path = file_path
        self.scheme = scheme

    def reset(self):
        """
        re-set all states
        """
        self.correct_labels = 0
        self.total_labels = 0

    def calc_acc_batch(self, decoded_data, target_data, *args):
        """
        update statics for accuracy

        args: 
            decoded_data (batch_size, seq_len): prediction sequence
            target_data (batch_size, seq_len): ground-truth
        """
        batch_decoded = torch.unbind(decoded_data, 1)
        batch_targets = torch.unbind(target_data, 0)

        for decoded, target in zip(batch_decoded, batch_targets):
            gold = target % len(self.l_map)
            # remove padding
            length = utils.find_length_from_labels(gold, self.l_map)
            gold = gold[:length].numpy()
            best_path = decoded[:length].numpy()

            self.total_labels += length
            self.correct_labels += np.sum(np.equal(best_path, gold))

    def write_result(self, decoded_data, target_data, feature, fout):

        batch_decoded = torch.unbind(decoded_data, 1)
        batch_targets = torch.unbind(target_data, 0)
        idx2item = self.r_l_map
        lines = list()
        for predict, target, sentence in zip(batch_decoded, batch_targets, feature):
            gold = target % len(self.l_map)
            length = utils.find_length_from_labels(gold, self.l_map)
            predict = predict[:length].numpy()
            gold = gold[:length].numpy()
            sentence = sentence.tokens[:length]
            for i in range(length):
                # lines.append(f'{sentence[i]} '
                #              f'{idx2item[predict[i]]} '
                #              f'{idx2item[gold[i]]}\n')
                fout.write(f'{sentence[i]} '
                           f'{idx2item[predict[i]]} '
                           f'{idx2item[gold[i]]} '
                           f'\n')
            fout.write('\n')

    def call_conlleval(self, prefix, scheme='BIOES'):
        file_path = self.file_path + f'/{prefix}.log'
        file_path_to = self.file_path + f'/{prefix}.BIO'
        if self.scheme == 'BIOES':
            tagSchemeConvert = subprocess.check_output(f'python tools/convertResultTagScheme.py {file_path} {file_path_to}',
                                                       shell=True,
                                                       timeout=200)
        else:
            file_path_to = file_path
        output = subprocess.check_output(f'perl tools/conlleval < {file_path_to}',
                                         shell=True,
                                         timeout=200).decode('utf-8')
        # if 'train' in prefix:
        if 'test' in prefix:
            # pass
            delete = subprocess.check_output(f'rm -rf {file_path_to} {file_path}',
                                             shell=True,
                                             timeout=200).decode('utf-8')
        else:
            delete = subprocess.check_output(f'rm -rf {file_path_to} {file_path}',
                                                 shell=True,
                                                 timeout=200).decode('utf-8')
        out = output.split('\n')[1]
        assert out.startswith('accuracy'), "Wrong lines"
        result = re.findall(r"\d+\.?\d*", out)
        return float(result[-1]), float(result[1]), float(result[2]), None

    def acc_score(self, *args):
        """
        calculate accuracy score based on statics
        """
        if 0 == self.total_labels:
            return 0.0
        accuracy = float(self.correct_labels) / self.total_labels
        return accuracy


class eval_w(eval_batch):
    """evaluation class for word level model (LSTM-CRF)

    args: 
        packer: provide method to convert target into original space [TODO: need to improve]
        l_map: dictionary for labels
        score_type: use f1score with using 'f'

    """

    def __init__(self, l_map, score_type, file_path, encoder, scheme='BIOES'):
        eval_batch.__init__(self, l_map, file_path, scheme)
        self.encoder = encoder
        self.scheme = scheme

        self.decoder = CRFDecode_vb(len(l_map), l_map['<START>'], l_map['<PAD>'])

        self.eval_method = score_type
        if 'f' in score_type:
            # self.eval_b = self.calc_f1_batch
            # self.calc_s = self.f1_score
            self.eval_b = self.write_result
            self.calc_s = self.call_conlleval
        else:
            self.eval_b = self.calc_acc_batch
            self.calc_s = self.acc_score

    def calc_score(self, ner_model, dataset_loader, file_prefix=None):
        """
        calculate score for pre-selected metrics

        args: 
            ner_model: LSTM-CRF model
            dataset_loader: loader class for test set
        """
        if 'f' in self.eval_method:
            fout = open(self.file_path + f'/{file_prefix}.log', 'w')
        else:
            fout = None

        with torch.no_grad():

            ner_model.eval()
            self.reset()
            for batch in itertools.chain.from_iterable(dataset_loader):
                # mask = mask.transpose(0, 1).cuda()

                tg, mask = repack(batch)
                # mask = mask.transpose(0, 1).cuda()
                if self.encoder == 'lstm':
                    score, _ = ner_model.forward(batch)
                elif self.encoder == 'transformer':
                    score, hidden = ner_model.forward(batch, mask)
                decoded = self.decoder.decode(score.data, mask.data)
                self.eval_b(decoded, tg, batch, fout)

        if 'f' in self.eval_method:
            fout.close()
        return self.calc_s(file_prefix)


class eval_sec_crf(eval_batch):
    """evaluation class for word level model (LSTM-CRF)

    args:
        packer: provide method to convert target into original space [TODO: need to improve]
        l_map: dictionary for labels
        score_type: use f1score with using 'f'

    """

    def __init__(self, packer, l_map, score_type):
        eval_batch.__init__(self, packer, l_map)

        self.decoder = CRFSECDecode_vb(len(l_map), l_map['<start>'], l_map['<pad>'])

        if 'f' in score_type:
            self.eval_b = self.calc_f1_batch
            self.calc_s = self.f1_score
        else:
            self.eval_b = self.calc_acc_batch
            self.calc_s = self.acc_score

    def calc_score(self, ner_model, dataset_loader):
        """
        calculate score for pre-selected metrics

        args:
            ner_model: LSTM-CRF model
            dataset_loader: loader class for test set
        """
        ner_model.eval()
        self.reset()

        for feature, tg, mask in itertools.chain.from_iterable(dataset_loader):
            fea_v, _, mask_v = self.packer.repack_vb(feature, tg, mask)
            scores, _ = ner_model(fea_v)
            decoded = self.decoder.decode(scores.data, mask_v.data)
            self.eval_b(decoded, tg)

        return self.calc_s()

class eval_wc(eval_batch):
    """evaluation class for LM-LSTM-CRF

    args: 
        packer: provide method to convert target into original space [TODO: need to improve]
        l_map: dictionary for labels
        score_type: use f1score with using 'f'

    """

    def __init__(self, packer, l_map, score_type):
        eval_batch.__init__(self, packer, l_map)

        self.decoder = CRFDecode_vb(len(l_map), l_map['<start>'], l_map['<pad>'])

        if 'f' in score_type:
            self.eval_b = self.calc_f1_batch
            self.calc_s = self.f1_score
        else:
            self.eval_b = self.calc_acc_batch
            self.calc_s = self.acc_score

    def calc_score(self, ner_model, dataset_loader):
        """
        calculate score for pre-selected metrics

        args: 
            ner_model: LM-LSTM-CRF model
            dataset_loader: loader class for test set
        """
        ner_model.eval()
        self.reset()

        for f_f, f_p, b_f, b_p, w_f, tg, mask_v, len_v in itertools.chain.from_iterable(dataset_loader):
            f_f, f_p, b_f, b_p, w_f, _, mask_v = self.packer.repack_vb(f_f, f_p, b_f, b_p, w_f, tg, mask_v, len_v)
            scores = ner_model(f_f, f_p, b_f, b_p, w_f)
            decoded = self.decoder.decode(scores.data, mask_v.data)
            self.eval_b(decoded, tg)

        return self.calc_s()

# softmax
class eval_softmax(eval_batch):
    """evaluation class for word level model (LSTM-SOFTMAX)

    args:
        packer: provide method to convert target into original space [TODO: need to improve]
        l_map: dictionary for labels
        score_type: use f1score with using 'f'

    """

    def __init__(self, l_map, score_type, file_path=None, scheme='BIOES'):
        eval_batch.__init__(self, l_map, file_path)
        self.pad = l_map['<PAD>']
        self.eval_method = score_type
        if 'f' in score_type:
            # self.eval_b = self.calc_f1_batch
            # self.calc_s = self.f1_score
            self.eval_b = self.write_result
            self.calc_s = self.call_conlleval
        else:
            self.eval_b = self.calc_acc_batch
            self.calc_s = self.acc_score

    def decode(self, scores, masks, pad_tag):
        _, tags = torch.max(scores, 2)
        masks = ~masks
        tags.masked_fill_(masks, pad_tag)

        return tags.cpu()

    def calc_score(self, ner_model, dataset_loader, file_prefix=None):
        """
        calculate score for pre-selected metrics

        args:
            ner_model: LSTM-CRF model
            dataset_loader: loader class for test set
        """
        if 'f' in self.eval_method:
            fout = open(self.file_path + f'/{file_prefix}.log', 'w')
        else:
            fout = None

        with torch.no_grad():
            ner_model.eval()
            self.reset()

            for batch in itertools.chain.from_iterable(dataset_loader):
                tg, mask = repack(batch)
                # mask = mask.transpose(0, 1).cuda()
                score, _ = ner_model.forward(batch)
                decoded = self.decode(score.data, mask.cuda().data, self.pad)
                self.eval_b(decoded, tg, batch, fout)

        if 'f' in self.eval_method:
            fout.close()
        return self.calc_s(file_prefix)

    def calc_acc_batch(self, decoded_data, target_data, *args):
        """
        update statics for accuracy

        args:
            decoded_data (batch_size, seq_len): prediction sequence
            target_data (batch_size, seq_len): ground-truth
        """
        batch_decoded = torch.unbind(decoded_data, 1)
        batch_targets = torch.unbind(target_data, 0)

        for decoded, target in zip(batch_decoded, batch_targets):
            gold = target
            # remove padding
            length = utils.find_length_from_softmax_labels(gold, self.l_map)
            gold = gold[:length].numpy()
            best_path = decoded[:length].numpy()

            self.total_labels += length
            self.correct_labels += np.sum(np.equal(best_path, gold))

    def write_result(self, decoded_data, target_data, feature, fout):

        batch_decoded = torch.unbind(decoded_data, 1)
        batch_targets = torch.unbind(target_data, 0)
        idx2item = self.r_l_map
        lines = list()
        for predict, target, sentence in zip(batch_decoded, batch_targets, feature):
            tokens = sentence.tokens
            gold = target % len(self.l_map)
            length = utils.find_length_from_softmax_labels(gold, self.l_map)
            predict = predict[:length].numpy()
            gold = gold[:length].numpy()
            tokens = tokens[:length]
            for i in range(length):
                # lines.append(f'{sentence[i]} '
                #              f'{idx2item[predict[i]]} '
                #              f'{idx2item[gold[i]]}\n')
                fout.write(f'{tokens[i]} '
                           f'{idx2item[predict[i]]} '
                           f'{idx2item[gold[i]]} '
                           f'\n')
            fout.write('\n')

