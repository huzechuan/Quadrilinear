import itertools
from functools import reduce
import numpy as np
import json

import torch
import torch.nn as nn
import torch.nn.init
from itertools import groupby
from torch.autograd import Variable
import math
import random
import time
import sys

zip = getattr(itertools, 'izip', zip)
LOGZERO = -np.inf
EMPTY = -1

def log_info(info):
    now_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f'{now_time} '
          f'{info} ')
    # sys.stdout.write(f'{now_time} '
    #                  f'{info} ')
    sys.stdout.flush()
    # print(f"\033[0;31m {now_time} ")
    # print(f'\033[0m {info}')

def to_scalar(var):
    """change the first element of a tensor to scalar
    """
    return var.view(-1).data.tolist()[0]


def argmax(vec):
    """helper function to calculate argmax of input vector at dimension 1
    """
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)


def log_sum_exp(vec, m_size):
    """
    calculate log of exp sum

    args:
        vec (batch_size, vanishing_dim, hidden_dim) : input tensor
        m_size : hidden_dim
    return:
        batch_size, hidden_dim
    """
    _, idx = torch.max(vec, 1)  # B * 1 * M
    max_score = torch.gather(vec, 1, idx.view(-1, 1, m_size)).view(-1, 1, m_size)  # B * M

    return max_score.view(-1, m_size) + torch.log(torch.sum(torch.exp(vec - max_score.expand_as(vec)), 1)).view(-1,
                                                                                                                m_size)  # B * M

def log_sum_exp_sec(vec, m_size):

    _, idx = torch.max(vec, 1) # B * 1 * M * M
    max_score = torch.gather(vec, 1, idx.view(-1, 1, m_size, m_size)).view(-1, 1, m_size, m_size)
    #TODO 这里可能要 DEBUG 一下，看看 max 对不对
    return max_score.view(-1, m_size, m_size) + torch.log(torch.sum(torch.exp(vec - max_score.expand_as(vec)), 1)).view(-1, m_size, m_size)


def encode2char_safe(input_lines, char_dict):
    """
    get char representation of lines

    args:
        input_lines (list of strings) : input corpus
        char_dict (dictionary) : char-level dictionary
    return:
        forw_lines
    """
    unk = char_dict['<u>']
    forw_lines = [list(map(lambda m: list(map(lambda t: char_dict.get(t, unk), m)), line)) for line in input_lines]
    return forw_lines


def concatChar(input_lines, char_dict):
    """
    concat char into string

    args:
        input_lines (list of list of char) : input corpus
        char_dict (dictionary) : char-level dictionary
    return:
        forw_lines
    """
    features = [[char_dict[' ']] + list(reduce(lambda x, y: x + [char_dict[' ']] + y, sentence)) + [char_dict['\n']] for
                sentence in input_lines]
    return features

def find_length_from_feats(feats, feat_to_ix):
    """
    find length of unpadded features based on feature
    """
    end_position = len(feats) - 1
    for position, feat in enumerate(feats):
        if feat.data[0] == feat_to_ix['<eof>']:
            end_position = position
            break
    return end_position + 1


def find_length_from_labels(labels, label_to_ix):
    """
    find length of unpadded features based on labels
    """
    end_position = len(labels) - 1
    for position, label in enumerate(labels):
        if label == label_to_ix['<PAD>']:
            end_position = position
            break
    return end_position

def find_length_from_softmax_labels(labels, label_to_ix):
    """
        find length of unpadded features based on labels
        """
    end_position = len(labels)
    for position, label in enumerate(labels):
        if label == label_to_ix['<PAD>']:
            end_position = position
            break
    return end_position


def revlut(lut):
    return {v: k for k, v in lut.items()}

def iobes_to_spans(sequence, lut, strict_iob2=False):
    """
    convert to iobes to span
    """
    iobtype = 2 if strict_iob2 else 1
    chunks = []
    current = None

    for i, y in enumerate(sequence):
        label = lut[y]

        if label.startswith('B-'):

            if current is not None:
                chunks.append('@'.join(current))
            current = [label.replace('B-', ''), '%d' % i]

        elif label.startswith('S-'):

            if current is not None:
                chunks.append('@'.join(current))
                current = None
            base = label.replace('S-', '')
            chunks.append('@'.join([base, '%d' % i]))

        elif label.startswith('I-'):

            if current is not None:
                base = label.replace('I-', '')
                if base == current[0]:
                    current.append('%d' % i)
                else:
                    chunks.append('@'.join(current))
                    if iobtype == 2:
                        print('Warning')
                    current = [base, '%d' % i]

            else:
                current = [label.replace('I-', ''), '%d' % i]
                if iobtype == 2:
                    print('Warning')

        elif label.startswith('E-'):

            if current is not None:
                base = label.replace('E-', '')
                if base == current[0]:
                    current.append('%d' % i)
                    chunks.append('@'.join(current))
                    current = None
                else:
                    chunks.append('@'.join(current))
                    if iobtype == 2:
                        print('Warning')
                    current = [base, '%d' % i]
                    chunks.append('@'.join(current))
                    current = None

            else:
                current = [label.replace('E-', ''), '%d' % i]
                if iobtype == 2:
                    print('Warning')
                chunks.append('@'.join(current))
                current = None
        else:
            if current is not None:
                chunks.append('@'.join(current))
            current = None

    if current is not None:
        chunks.append('@'.join(current))

    return set(chunks)

def init_embedding(input_embedding):
    """
    Initialize embedding
    """
    bias = np.sqrt(3.0 / input_embedding.size(1))
    nn.init.uniform_(input_embedding, -bias, bias)


def init_linear(input_linear):
    """
    Initialize linear transformation
    """
    bias = np.sqrt(6.0 / (input_linear.weight.size(0) + input_linear.weight.size(1)))
    nn.init.uniform_(input_linear.weight, -bias, bias)
    if input_linear.bias is not None:
        input_linear.bias.data.zero_()

def init_trans(trans):
    """Initialize linear transformation
    """
    bias = np.sqrt(6.0 / (trans.size(0) + trans.size(1)))
    nn.init.uniform_(trans, -bias, bias)

def init_tensor(tens, std=0.1545):
    """Initialize linear transformation
        """
    bias = std
    # bias = 0.1545# np.sqrt(6.0 / (tens.size(0) + tens.size(1)))
    nn.init.normal_(tens, 0, bias)

def rand_emb(embedding):
    bias = np.sqrt(3.0 / embedding.size(0))
    nn.init.uniform_(embedding, -bias, bias)
    return embedding

def init_lstm(input_lstm):
    """
    Initialize lstm
    """
    for ind in range(0, input_lstm.num_layers):
        weight = eval('input_lstm.weight_ih_l' + str(ind))
        bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
        nn.init.uniform(weight, -bias, bias)
        weight = eval('input_lstm.weight_hh_l' + str(ind))
        bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
        nn.init.uniform(weight, -bias, bias)

    if input_lstm.bias:
        for ind in range(0, input_lstm.num_layers):
            weight = eval('input_lstm.bias_ih_l' + str(ind))
            weight.data.zero_()
            weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
            weight = eval('input_lstm.bias_hh_l' + str(ind))
            weight.data.zero_()
            weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1


def checknan(a):
    bol = torch.isnan(a)
    sum = torch.sum(bol)
    if sum.cpu().numpy() != 0:
        print('\n--------------------------------------------------------------------------\n')
        return False
    return True


def checkinf(b):
    bol = torch.isinf(b)
    sum = torch.sum(bol)
    if sum.cpu().numpy() != 0:
        print('\n--------------------------------------------------------------------------\n')
        return False
    return True
