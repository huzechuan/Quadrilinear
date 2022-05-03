"""
.. module:: lstm_crf
    :synopsis: lstm_crf

.. moduleauthor:: Liyuan Liu
"""

import torch
import torch.autograd as autograd
import torch.nn as nn
import models
import numpy as np
from models.crf import (
    CRF,
    CRF_TRI
)
# import Tri_Linear.models.utils as utils


def init_lstm(input_lstm):
    """
    Initialize lstm
    """
    for ind in range(0, input_lstm.num_layers):
        weight = eval('input_lstm.weight_ih_l' + str(ind))
        bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
        nn.init.uniform_(weight, -bias, bias)
        weight = eval('input_lstm.weight_hh_l' + str(ind))
        bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
        nn.init.uniform_(weight, -bias, bias)

    if input_lstm.bias:
        for ind in range(0, input_lstm.num_layers):
            weight = eval('input_lstm.bias_ih_l' + str(ind))
            weight.data.zero_()
            weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
            weight = eval('input_lstm.bias_hh_l' + str(ind))
            weight.data.zero_()
            weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1


class LSTM_CRF(nn.Module):
    """LSTM_CRF model

    args:
        vocab_size: size of word dictionary
        tagset_size: size of label set
        embedding_dim: size of word embedding
        hidden_dim: size of word-level blstm hidden dim
        rnn_layers: number of word-level lstm layers
        dropout_ratio: dropout ratio
        large_CRF: use CRF_L or not, refer model.crf.CRF_L and model.crf.CRF_S for more details
    """

    def __init__(self, tagset_size, embeddings, hidden_dim, rnn_layers, dropout_ratio,
                 trilinear=False, dim=200):
        super(LSTM_CRF, self).__init__()
        self.embeddings = embeddings
        self.embedding_dim = embeddings.embedding_length
        self.hidden_dim = hidden_dim
        # self.vocab_size = vocab_size
        #
        # self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim, hidden_dim // 2,
                            num_layers=rnn_layers, bidirectional=True)
        self.rnn_layers = rnn_layers

        self.dropout1 = nn.Dropout(p=dropout_ratio)
        self.dropout2 = nn.Dropout(p=dropout_ratio)

        self.tagset_size = tagset_size
        if trilinear:
            self.crf = CRF_TRI(hidden_dim, tagset_size, dim)
        else:
            self.crf = CRF(hidden_dim, tagset_size)

        self.batch_size = 1
        self.seq_length = 1

    def set_batch_size(self, bsize):
        """
        set batch size
        """
        self.batch_size = bsize

    def set_batch_seq_size(self, sentence):
        """
        set batch size and sequence length
        """
        tmp = sentence.size()
        self.seq_length = tmp[0]
        self.batch_size = tmp[1]


    def rand_init(self, init_embedding=False):
        """
        random initialization

        args:
            init_embedding: random initialize embedding or not
        """

        init_lstm(self.lstm)
        self.crf.rand_init()

    def load_pretrained_embedding(self, pre_embeddings):
        """
        load pre-trained word embedding

        args:
            pre_word_embeddings (self.word_size, self.word_dim) : pre-trained embedding
        """
        assert (pre_embeddings.size()[1] == self.embedding_dim)
        self.word_embeds.weight = nn.Parameter(pre_embeddings)

    def forward(self, feats, hidden=None):
        '''
        args:
            sentence (word_seq_len, batch_size) : word-level representation of sentence
            hidden: initial hidden state

        return:
            crf output (word_seq_len, batch_size, tag_size, tag_size), hidden
        '''
        embeds = self.embeddings.embed(feats)
        self.set_batch_seq_size(embeds)

        d_embeds = self.dropout1(embeds)

        lstm_out, hidden = self.lstm(d_embeds, hidden)
        # lstm_out = lstm_out.view(-1, self.hidden_dim)

        d_lstm_out = self.dropout2(lstm_out)

        crf_out = self.crf(d_lstm_out)

        crf_out = crf_out.view(self.seq_length, self.batch_size, self.tagset_size, self.tagset_size)

        return crf_out, hidden


class LSTM_SEC_CRF(nn.Module):
    """LSTM_SEC_ORDER_CRF model
    author: huzechuan@std.uestc.edu.cn
    args:
        vocab_size: size of word dictionary
        tagset_size: size of label set
        embedding_dim: size of word embedding
        hidden_dim: size of word-level blstm hidden dim
        rnn_layers: number of word-level lstm layers
        dropout_ratio: dropout ratio
    """

    def __init__(self, vocab_size, tagset_size, embedding_dim, hidden_dim, rnn_layers, dropout_ratio):
        super(LSTM_SEC_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=rnn_layers, bidirectional=True)
        self.rnn_layers = rnn_layers

        self.dropout1 = nn.Dropout(p=dropout_ratio)
        self.dropout2 = nn.Dropout(p=dropout_ratio)

        self.tagset_size = tagset_size

        self.crf = crf.CRF_S_SEC(hidden_dim, tagset_size)

        self.batch_size = 1
        self.seq_length = 1

    def rand_init_hidden(self):
        """
        random initialize hidden variable
        """
        return autograd.Variable(
            torch.randn(2 * self.rnn_layers, self.batch_size, self.hidden_dim // 2)), autograd.Variable(
            torch.randn(2 * self.rnn_layers, self.batch_size, self.hidden_dim // 2))

    def set_batch_size(self, bsize):
        """
        set batch size
        """
        self.batch_size = bsize

    def set_batch_seq_size(self, sentence):
        """
        set batch size and sequence length
        """
        tmp = sentence.size()
        self.seq_length = tmp[0]
        self.batch_size = tmp[1]

    def load_pretrained_embedding(self, pre_embeddings):
        """
        load pre-trained word embedding

        args:
            pre_word_embeddings (self.word_size, self.word_dim) : pre-trained embedding
        """
        assert (pre_embeddings.size()[1] == self.embedding_dim)
        self.word_embeds.weight = nn.Parameter(pre_embeddings)



    def rand_init(self, init_embedding=False):
        """
        random initialization

        args:
            init_embedding: random initialize embedding or not
        """
        utils.init_lstm(self.lstm)
        self.crf.rand_init()

    def forward(self, sentence, hidden=None):
        '''
        args:
            sentence (word_seq_len, batch_size) : word-level representation of sentence
            hidden: initial hidden state

        return:
            crf output (word_seq_len, batch_size, tag_size, tag_size), hidden
        '''
        self.set_batch_seq_size(sentence)

        embeds = self.word_embeds(sentence)
        d_embeds = self.dropout1(embeds)

        lstm_out, hidden = self.lstm(d_embeds, hidden)
        lstm_out = lstm_out.view(-1, self.hidden_dim)

        d_lstm_out = self.dropout2(lstm_out)

        crf_out = self.crf(d_lstm_out)
        crf_out = crf_out.view(self.seq_length, self.batch_size, self.tagset_size, self.tagset_size,
                               self.tagset_size)

        return crf_out, hidden
