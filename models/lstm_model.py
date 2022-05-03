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
    CRF_TRI,
    CRFLoss_vb
)
import models.utils as utils
import time

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


class LSTM_Model(nn.Module):
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

    def __init__(self, tag_map, embeddings, hidden_dim, rnn_layers, dropout_ratio,
                 use_crf=True, tri_parameter=None):
        super(LSTM_Model, self).__init__()
        self.embeddings = embeddings
        self.embedding_dim = embeddings.embedding_length
        self.hidden_dim = hidden_dim
        self.use_crf = use_crf
        self.tag_map = tag_map
        # self.vocab_size = vocab_size
        #
        # self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim, hidden_dim // 2,
                            num_layers=rnn_layers, bidirectional=True)
        self.rnn_layers = rnn_layers

        self.dropout1 = nn.Dropout(p=dropout_ratio)
        self.dropout2 = nn.Dropout(p=dropout_ratio)

        self.tagset_size = len(tag_map)
        if use_crf:
            self.top_layer = CRF(hidden_dim, self.tagset_size, tri_parameter)
            self.loss = CRFLoss_vb(self.tagset_size, self.tag_map['<START>'], self.tag_map['<PAD>'])
        else:
            self.top_layer = nn.Linear(hidden_dim, self.tagset_size, bias=True)
            self.loss = nn.CrossEntropyLoss(reduction='sum')

        self.batch_size = None
        self.seq_length = None

        self.lstm_time = 0
        self.top_time = 0

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
        if self.use_crf:
            self.top_layer.rand_init()

    def load_pretrained_embedding(self, pre_embeddings):
        """
        load pre-trained word embedding

        args:
            pre_word_embeddings (self.word_size, self.word_dim) : pre-trained embedding
        """
        assert (pre_embeddings.size()[1] == self.embedding_dim)
        self.word_embeds.weight = nn.Parameter(pre_embeddings)

    def crit(self, scores, tags, masks):
        if self.use_crf:
            loss = self.loss(scores, tags, masks)
        else:
            # Sentences averaged version
            # print(tags.size())
            tags = tags.contiguous().view(-1, )
            scores = scores.view(-1, self.tagset_size)
            scores = scores.masked_select(masks.contiguous().view(-1, 1).expand(-1, self.tagset_size)).view(-1, self.tagset_size)
            masks = masks.contiguous().view(-1, )
            tags = tags.masked_select(masks)
            loss = self.loss(scores, tags)
            loss = loss / self.batch_size

        # else:
        #     # Token averaged version
        #     loss = 0
        #     scores = scores.transpose(0, 1)
        #     tags = tags.transpose(0, 1)
        #     masks = masks.transpose(0, 1)
        #     for score, tag, mask in zip(scores, tags, masks):
        #         tag = tag.view(-1, )
        #         score = score.view(-1, self.tagset_size)
        #         score = score.masked_select(mask.view(-1, 1).expand(-1, self.tagset_size)).view(-1, self.tagset_size)
        #         mask = mask.view(-1, )
        #         tag = tag.masked_select(mask)
        #         loss += self.loss(score, tag)

        return loss #/ self.batch_size

    def forward(self, feats, hidden=None):
        '''
        args:
            sentence (word_seq_len, batch_size) : word-level representation of sentence
            hidden: initial hidden state

        return:
            crf output (word_seq_len, batch_size, tag_size, tag_size), hidden
        '''
        # lstm_start = time.time()

        embeds = self.embeddings.embed(feats)
        self.set_batch_seq_size(embeds)

        d_embeds = self.dropout1(embeds)

        lstm_out, hidden = self.lstm(d_embeds, hidden)
        # lstm_out = lstm_out.view(-1, self.hidden_dim)

        d_lstm_out = self.dropout2(lstm_out)

        # self.lstm_time += time.time() - lstm_start

        # top_start = time.time()
        score = self.top_layer(d_lstm_out)

        # self.top_time += time.time() - top_start

        if self.use_crf:
            score = score.view(self.seq_length, self.batch_size, self.tagset_size, self.tagset_size)
        else:
            score = score.view(self.seq_length, self.batch_size, self.tagset_size)

        return score, hidden


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


class TransformerEncoder(nn.Module):
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

    def __init__(self, tag_map, embeddings, hidden_dim, rnn_layers, dropout_ratio,
                 use_crf=True, tri_parameter=None):
        super(TransformerEncoder, self).__init__()
        self.embeddings = embeddings
        self.embedding_dim = embeddings.embedding_length
        self.hidden_dim = self.embedding_dim
        self.use_crf = use_crf
        self.tag_map = tag_map
        # self.vocab_size = vocab_size
        #
        # self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        # self.lstm = nn.LSTM(self.embedding_dim, hidden_dim // 2,
        #                     num_layers=rnn_layers, bidirectional=True)
        # self.transformer_encoder = nn.MultiheadAttention(self.embedding_dim, 1, dropout=0.1)
        encoder_layer = nn.TransformerEncoderLayer(self.embedding_dim, nhead=1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.rnn_layers = rnn_layers

        self.dropout1 = nn.Dropout(p=dropout_ratio)
        self.dropout2 = nn.Dropout(p=dropout_ratio)

        self.tagset_size = len(tag_map)
        if use_crf:
            self.top_layer = CRF(self.hidden_dim, self.tagset_size, tri_parameter)
            self.loss = CRFLoss_vb(self.tagset_size, self.tag_map['<START>'], self.tag_map['<PAD>'])
        else:
            self.top_layer = nn.Linear(self.hidden_dim, self.tagset_size, bias=True)
            self.loss = nn.CrossEntropyLoss(reduction='sum')

        self.batch_size = None
        self.seq_length = None

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

        # init_lstm(self.lstm)
        if self.use_crf:
            self.top_layer.rand_init()

    def load_pretrained_embedding(self, pre_embeddings):
        """
        load pre-trained word embedding

        args:
            pre_word_embeddings (self.word_size, self.word_dim) : pre-trained embedding
        """
        assert (pre_embeddings.size()[1] == self.embedding_dim)
        self.word_embeds.weight = nn.Parameter(pre_embeddings)

    def crit(self, scores, tags, masks):
        if self.use_crf:
            loss = self.loss(scores, tags, masks)
        else:
            # Sentences averaged version
            tags = tags.view(-1, )
            scores = scores.view(-1, self.tagset_size)
            scores = scores.masked_select(masks.view(-1, 1).expand(-1, self.tagset_size)).view(-1, self.tagset_size)
            masks = masks.view(-1, )
            tags = tags.masked_select(masks)
            loss = self.loss(scores, tags)
            loss = loss / self.batch_size

        return loss #/ self.batch_size

    def forward(self, feats, mask=None, hidden=None):
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

        # lstm_out, hidden = self.lstm(d_embeds, hidden)
        # att_out, att_weight = self.transformer_encoder(d_embeds, d_embeds, d_embeds, ~mask.transpose(0, 1)) #, mask.transpose(0, 1)
        att_out = self.transformer_encoder(d_embeds, src_key_padding_mask=~mask.transpose(0, 1)) #, mask.transpose(0, 1)
        # lstm_out = lstm_out.view(-1, self.hidden_dim)

        # d_lstm_out = self.dropout2(lstm_out)
        d_lstm_out = self.dropout2(att_out)

        score = self.top_layer(d_lstm_out)

        if self.use_crf:
            score = score.view(self.seq_length, self.batch_size, self.tagset_size, self.tagset_size)
        else:
            score = score.view(self.seq_length, self.batch_size, self.tagset_size)

        return score, hidden
        # return att_out, hidden