"""
.. module:: crf
    :synopsis: conditional random field

.. moduleauthor:: Liyuan Liu

.. modified by: huzechuan@std.uestc.edu.cn
.. add items: second order crf module
"""

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.sparse as sparse
import models.utils as utils
from models.utils import log_info
import math


class CRF(nn.Module):
    """Conditional Random Field (CRF) layer. This version is used in Lample et al. 2016, has less parameters than CRF_L.

    args:
        hidden_dim: input dim size
        tagset_size: target_set_size
        if_biase: whether allow bias in linear trans

    """

    def __init__(self, hidden_dim, tagset_size, tri_parameter=None, if_bias=True):
        super(CRF, self).__init__()
        self.tagset_size = tagset_size
        self.use_which = tri_parameter['use_which']
        log_info(tri_parameter['std'])
        if self.use_which == 'Trilinear':
            self.dense = TriLinearScore(hidden_dim,
                                        self.tagset_size,
                                        tri_parameter['tag_dim'],
                                        tri_parameter['rank'],
                                        tri_parameter['std'])
        elif self.use_which == 'Bilinear':
            self.dense = BiLinearScore(hidden_dim,
                                       self.tagset_size,
                                       tri_parameter['tag_dim'],
                                       tri_parameter['std'])
        elif self.use_which == 'FullTrilinear':
            self.dense = FullTriLinearScore(hidden_dim,
                                            self.tagset_size,
                                            tri_parameter['tag_dim'],
                                            tri_parameter['std'])
        elif self.use_which == 'ThreeBilinear':
            self.dense = ThreeBiLinearScore(hidden_dim,
                                            self.tagset_size,
                                            tri_parameter['tag_dim'],
                                            tri_parameter['std'])
        elif self.use_which == 'Qualinear':
            self.dense = QuaLinearScore(hidden_dim,
                                        self.tagset_size,
                                        tri_parameter['tag_dim'],
                                        tri_parameter['rank'],
                                        tri_parameter['std'],
                                        neighbor=tri_parameter['neighbor'],
                                        normalization=tri_parameter['normalize'])
        elif self.use_which == 'Pentalinear':
            self.dense = PentaLinearScore(hidden_dim,
                                          self.tagset_size,
                                          tri_parameter['tag_dim'],
                                          tri_parameter['rank'],
                                          tri_parameter['std'])
        elif self.use_which is not None and 'ConcatScore' == self.use_which.split('-')[0]:
            self.dense = ConcatScore(hidden_dim,
                                     self.tagset_size,
                                     tri_parameter['tag_dim'],
                                     tri_parameter['rank'],
                                     tri_parameter['std'],
                                     self.use_which.split('-')[-1])
        else:
            self.dense = CRFScore(hidden_dim, self.tagset_size, if_bias)
            # self.transitions = nn.Parameter(torch.Tensor(self.tagset_size, self.tagset_size))

    def rand_init(self):
        #     """random initialization
        #     """
        #     if self.use_which == 'Linear':
        #         utils.init_linear(self.dense)
        #         self.transitions.data.zero_()
        self.dense.rand_init()

    def rand_test(self):
        self.transitions.data.zero_()

    def forward(self, feats):
        """
        args:
            feats (batch_size, seq_len, hidden_dim) : input score from previous layers
        return:
            output from crf layer ( (batch_size * seq_len), tag_size, tag_size)
        """
        # if self.use_which is 'Linear':
        #     scores = self.dense(feats).view(-1, 1, self.tagset_size)
        #     ins_num = scores.size(0)
        #     crf_scores = scores.expand(ins_num, self.tagset_size, self.tagset_size) + self.transitions.view(1,
        #                                                                                                     self.tagset_size,
        #                                                                                                     self.tagset_size).expand(
        #         ins_num, self.tagset_size, self.tagset_size)
        # else:
        crf_scores = self.dense(feats).view(-1, self.tagset_size, self.tagset_size)

        # tag_emb = torch.mm(self.trans, self.W)
        # transitions = torch.einsum('ak, bk->ab', tag_emb, tag_emb)
        # crf_scores = scores.expand(ins_num, self.tagset_size, self.tagset_size) + transitions.view(1, self.tagset_size, self.tagset_size).expand(ins_num, self.tagset_size, self.tagset_size)

        return crf_scores


class CRFScore(nn.Module):
    """
    Compute CRF scores which is emission + transition
    """

    def __init__(self, hidden_dim, tagset_size, if_bias=True):
        super(CRFScore, self).__init__()
        self.hidden_dim = hidden_dim
        self.tagset_size = tagset_size

        self.linear_layer = nn.Linear(hidden_dim, self.tagset_size, bias=if_bias)
        self.transitions = nn.Parameter(torch.Tensor(self.tagset_size, self.tagset_size))

        # self.rand_init()

    def rand_init(self):
        utils.init_linear(self.linear_layer)
        self.transitions.data.zero_()

    def forward(self, word_features):
        scores = self.linear_layer(word_features).view(-1, 1, self.tagset_size)
        ins_num = scores.size(0)
        scores = scores.expand(ins_num, self.tagset_size, self.tagset_size)

        crf_scores = scores + self.transitions.view(1, self.tagset_size, self.tagset_size).expand(ins_num,
                                                                                                  self.tagset_size,
                                                                                                  self.tagset_size)
        return crf_scores


class QuaLinearScore(nn.Module):
    """
    Outer product version of trilinear function.

    Trilinear attention layer.
    """

    def __init__(self, wemd_size, tagset_size, temd_size=20, rank=396, std=0.1545, neighbor='prev', normalization=False, **kwargs):
        """
        Args:
            input_size_encoder: int
                the dimension of the encoder input.
            input_size_decoder: int
                the dimension of the decoder input.
            num_labels: int
                the number of labels of the crf layer
            biaffine: bool
                if apply bi-affine parameter.
            **kwargs:
        """
        super(QuaLinearScore, self).__init__()
        self.wemd_size = wemd_size
        self.tagset_size = tagset_size
        self.temd_size = temd_size
        self.rank = rank
        self.std = std

        self.tag_emd = nn.Parameter(torch.Tensor(self.tagset_size, self.temd_size))
        self.T = nn.Parameter(torch.Tensor(self.wemd_size, self.rank))
        self.U = nn.Parameter(torch.Tensor(self.wemd_size, self.rank))
        self.V = nn.Parameter(torch.Tensor(self.temd_size, self.rank))
        self.W = nn.Parameter(torch.Tensor(self.temd_size, self.rank))
        self.first_wemb = nn.Parameter(torch.Tensor(1, self.wemd_size))
        self.last_wemb = nn.Parameter(torch.Tensor(1, self.wemd_size))

        self.neighbor = neighbor
        self.normalization = normalization

        if self.neighbor == 'prev':
            log_info('using previous word for quadrilinear.')
        elif self.neighbor == 'next':
            log_info('using next word for quadrilinear.')
        elif self.neighbor == 'mean':
            log_info('using mean of next word and prev word for quadrilinear.')

        if self.normalization:
            log_info('using normalization for quadrilinear.')
        else:
            log_info('not using normalization for quadrilinear.')


        # self.rand_init()

    def rand_init(self):
        '''random initialization
        '''

        utils.init_trans(self.tag_emd)
        utils.init_embedding(self.first_wemb)
        utils.init_tensor(self.T, self.std)
        utils.init_tensor(self.U, self.std)
        utils.init_tensor(self.V, self.std)
        utils.init_tensor(self.W, self.std)
        # std = 1.0
        # nn.init.xavier_normal_(self.tag_emd)
        # nn.init.xavier_normal_(self.U, gain=std)
        # nn.init.xavier_normal_(self.V, gain=std)
        # nn.init.xavier_normal_(self.W, gain=std)

    def forward(self, word_emd):
        """
        Args:

        Returns: Tensor
            the energy tensor with shape = [batch, num_label, length, length]
        """
        assert word_emd.size(2) == self.wemd_size, 'batch sizes of encoder and decoder are requires to be equal.'
        if self.neighbor == 'prev':
            first_word = self.first_wemb.view(1, 1, self.wemd_size).expand(1, word_emd.size(1), word_emd.size(2))
            prev_word_emd = word_emd[:word_emd.size(0) - 1, :, :]#.clone()
            neighbor_word_emd = torch.cat([first_word, prev_word_emd], 0)
        elif self.neighbor == 'next':
            last_word = self.last_wemb.view(1, 1, self.wemd_size).expand(1, word_emd.size(1), word_emd.size(2))
            next_word_emd = word_emd[1:, :, :]  # .clone()
            neighbor_word_emd = torch.cat([next_word_emd, last_word], 0)
        elif self.neighbor == 'mean':
            first_word = self.first_wemb.view(1, 1, self.wemd_size).expand(1, word_emd.size(1), word_emd.size(2))
            prev_word_emd = word_emd[:word_emd.size(0) - 1, :, :]  # .clone()
            last_word = self.last_wemb.view(1, 1, self.wemd_size).expand(1, word_emd.size(1), word_emd.size(2))
            next_word_emd = word_emd[1:, :, :]  # .clone()
            prev_word_emd = torch.cat([first_word, prev_word_emd], 0)
            next_word_emd = torch.cat([next_word_emd, last_word], 0)
            neighbor_word_emd = (prev_word_emd + next_word_emd) / 2
        # print(f'{word_emd1.size()}, {last_word.size()}, {word_emd.size()}')
        # (n x m x d) * (d x k) -> (n x m x k)
        g0 = torch.matmul(word_emd, self.U)
        g1 = torch.matmul(neighbor_word_emd, self.T)
        g2 = torch.matmul(self.tag_emd, self.V)
        g3 = torch.matmul(self.tag_emd, self.W)
        # print(f'{g0.size}: {g1.size()}')
        temp01 = g0 * g1
        temp012 = torch.einsum('nak, bk->nabk', [temp01, g2])
        score = torch.einsum('nabk, ck->nabc', [temp012, g3])
        if self.normalization:
            score = score / math.sqrt(self.rank)
        return score


class TriLinearScore(nn.Module):
    """
    Outer product version of trilinear function.

    Trilinear attention layer.
    """

    def __init__(self, wemd_size, tagset_size, temd_size=20, rank=396, std=0.1545, **kwargs):
        """
        Args:
            input_size_encoder: int
                the dimension of the encoder input.
            input_size_decoder: int
                the dimension of the decoder input.
            num_labels: int
                the number of labels of the crf layer
            biaffine: bool
                if apply bi-affine parameter.
            **kwargs:
        """
        super(TriLinearScore, self).__init__()
        self.wemd_size = wemd_size
        self.tagset_size = tagset_size
        self.temd_size = temd_size
        self.rank = rank
        self.std = std

        self.tag_emd = nn.Parameter(torch.Tensor(self.tagset_size, self.temd_size))
        self.U = nn.Parameter(torch.Tensor(self.wemd_size, self.rank))
        self.V = nn.Parameter(torch.Tensor(self.temd_size, self.rank))
        self.W = nn.Parameter(torch.Tensor(self.temd_size, self.rank))

        # self.rand_init()

    def rand_init(self):
        '''random initialization
        '''

        utils.init_trans(self.tag_emd)
        utils.init_tensor(self.U, self.std)
        utils.init_tensor(self.V, self.std)
        utils.init_tensor(self.W, self.std)
        # std = 1.0
        # nn.init.xavier_normal_(self.tag_emd)
        # nn.init.xavier_normal_(self.U, gain=std)
        # nn.init.xavier_normal_(self.V, gain=std)
        # nn.init.xavier_normal_(self.W, gain=std)

    def forward(self, word_emd):
        """
        Args:

        Returns: Tensor
            the energy tensor with shape = [batch, num_label, length, length]
        """
        assert word_emd.size(2) == self.wemd_size, 'batch sizes of encoder and decoder are requires to be equal.'

        # (n x m x d) * (d x k) -> (n x m x k)
        g0 = torch.matmul(word_emd, self.U)
        g1 = torch.matmul(self.tag_emd, self.V)
        g2 = torch.matmul(self.tag_emd, self.W)
        temp12 = torch.einsum('nak,bk->nabk', (g0, g1))
        score = torch.einsum('nabk,ck->nabc', (temp12, g2))

        return score


class BiLinearScore(nn.Module):
    """
    Outer product version of trilinear function.

    Trilinear attention layer.
    """

    def __init__(self, wemd_size, tagset_size, temd_size=20, std=0.1545, **kwargs):
        """
        Args:
            input_size_encoder: int
                the dimension of the encoder input.
            input_size_decoder: int
                the dimension of the decoder input.
            num_labels: int
                the number of labels of the crf layer
            biaffine: bool
                if apply bi-affine parameter.
            **kwargs:
        """
        super(BiLinearScore, self).__init__()
        self.wemd_size = wemd_size
        self.tagset_size = tagset_size
        self.temd_size = temd_size
        self.std = std

        self.tag_emd = nn.Parameter(torch.Tensor(self.tagset_size, self.temd_size))
        self.U = nn.Parameter(torch.Tensor(self.wemd_size, self.temd_size))
        self.V = nn.Parameter(torch.Tensor(self.temd_size, self.temd_size))

        # self.rand_init()

    def rand_init(self):
        '''random initialization
        '''

        utils.init_trans(self.tag_emd)
        utils.init_tensor(self.U, self.std)
        utils.init_tensor(self.V, self.std)
        # std = 1.0
        # nn.init.xavier_normal_(self.tag_emd)
        # nn.init.xavier_normal_(self.U, gain=std)
        # nn.init.xavier_normal_(self.V, gain=std)
        # nn.init.xavier_normal_(self.W, gain=std)

    def forward(self, word_emd):
        """
        Args:

        Returns: Tensor
            the energy tensor with shape = [batch, num_label, length, length]
        """
        assert word_emd.size(2) == self.wemd_size, 'batch sizes of encoder and decoder are requires to be equal.'

        # (n x m x d) * (d x k) -> (n x m x k)
        emissions = torch.einsum('nak, kd, td->nat', (word_emd, self.U, self.tag_emd))
        transitions = torch.einsum('ni, ij, tj->nt', (self.tag_emd, self.V, self.tag_emd))
        emissions = emissions.view(-1, 1, self.tagset_size)
        ins_num = emissions.size(0)
        crf_scores = emissions.expand(ins_num, self.tagset_size, self.tagset_size) + transitions.view(1,
                                                                                                      self.tagset_size,
                                                                                                      self.tagset_size).expand(
            ins_num, self.tagset_size, self.tagset_size)
        return crf_scores


class ThreeBiLinearScore(nn.Module):
    """
    Outer product version of trilinear function.

    Trilinear attention layer.
    """

    def __init__(self, wemd_size, tagset_size, temd_size=20, std=0.1545, **kwargs):
        """
        Args:
            input_size_encoder: int
                the dimension of the encoder input.
            input_size_decoder: int
                the dimension of the decoder input.
            num_labels: int
                the number of labels of the crf layer
            biaffine: bool
                if apply bi-affine parameter.
            **kwargs:
        """
        super(ThreeBiLinearScore, self).__init__()
        self.wemd_size = wemd_size
        self.tagset_size = tagset_size
        self.temd_size = temd_size
        self.std = std

        self.tag_emd = nn.Parameter(torch.Tensor(self.tagset_size, self.temd_size))
        self.U = nn.Parameter(torch.Tensor(self.wemd_size, self.temd_size))
        self.V = nn.Parameter(torch.Tensor(self.wemd_size, self.temd_size))
        self.W = nn.Parameter(torch.Tensor(self.temd_size, self.temd_size))

        # self.rand_init()

    def rand_init(self):
        '''random initialization
        '''

        utils.init_trans(self.tag_emd)
        utils.init_tensor(self.U, self.std)
        utils.init_tensor(self.V, self.std)
        utils.init_tensor(self.W, self.std)
        # std = 1.0
        # nn.init.xavier_normal_(self.tag_emd)
        # nn.init.xavier_normal_(self.U, gain=std)
        # nn.init.xavier_normal_(self.V, gain=std)
        # nn.init.xavier_normal_(self.W, gain=std)

    def forward(self, word_emd):
        """
        Args:

        Returns: Tensor
            the energy tensor with shape = [batch, num_label, length, length]
        """
        assert word_emd.size(2) == self.wemd_size, 'batch sizes of encoder and decoder are requires to be equal.'

        # (n x m x d) * (d x k) -> (n x m x k)
        emissions = torch.einsum('nak, kd, td->nat', (word_emd, self.U, self.tag_emd))
        emissions1 = torch.einsum('nak, kd, td->nat', (word_emd, self.V, self.tag_emd))
        transitions = torch.einsum('ni, ij, tj->nt', (self.tag_emd, self.W, self.tag_emd))

        emissions = emissions.view(-1, 1, self.tagset_size)
        ins_num = emissions.size(0)
        emissions = emissions.expand(ins_num, self.tagset_size, self.tagset_size)
        emissions1 = emissions1.view(-1, 1, self.tagset_size).expand(ins_num, self.tagset_size, self.tagset_size)
        transitions = transitions.view(1, self.tagset_size, self.tagset_size).expand(ins_num, self.tagset_size,
                                                                                     self.tagset_size)

        crf_scores = emissions + emissions1 + transitions

        return crf_scores


class FullTriLinearScore(nn.Module):
    """
    Outer product version of trilinear function.

    Trilinear attention layer.
    """

    def __init__(self, wemd_size, tagset_size, temd_size=20, std=0.1545, **kwargs):
        """
        Args:
            input_size_encoder: int
                the dimension of the encoder input.
            input_size_decoder: int
                the dimension of the decoder input.
            num_labels: int
                the number of labels of the crf layer
            biaffine: bool
                if apply bi-affine parameter.
            **kwargs:
        """
        super(FullTriLinearScore, self).__init__()
        self.wemd_size = wemd_size
        self.tagset_size = tagset_size
        self.temd_size = temd_size
        self.std = std

        self.tag_emd = nn.Parameter(torch.Tensor(self.tagset_size, self.temd_size))
        self.W = nn.Parameter(torch.Tensor(self.wemd_size, self.temd_size, self.temd_size))

        # self.rand_init()

    def rand_init(self):
        '''random initialization
        '''

        utils.init_trans(self.tag_emd)
        utils.init_tensor(self.W, self.std)
        # std = 1.0
        # nn.init.xavier_normal_(self.tag_emd)
        # nn.init.xavier_normal_(self.U, gain=std)
        # nn.init.xavier_normal_(self.V, gain=std)
        # nn.init.xavier_normal_(self.W, gain=std)

    def forward(self, word_emd):
        """
        Args:

        Returns: Tensor
            the energy tensor with shape = [batch, num_label, length, length]
        """
        assert word_emd.size(2) == self.wemd_size, 'batch sizes of encoder and decoder are requires to be equal.'

        # (n x m x d) * (d x k) -> (n x m x k)
        score = torch.einsum('nsx, xgh, yg, zh->nsyz',
                             (word_emd,
                              self.W,
                              self.tag_emd,
                              self.tag_emd))

        return score


class PentaLinearScore(nn.Module):
    """
    Outer product version of trilinear function.

    Trilinear attention layer.
    """

    def __init__(self, wemd_size, tagset_size, temd_size=20, rank=396, std=0.1545, **kwargs):
        """
        Args:
            input_size_encoder: int
                the dimension of the encoder input.
            input_size_decoder: int
                the dimension of the decoder input.
            num_labels: int
                the number of labels of the crf layer
            biaffine: bool
                if apply bi-affine parameter.
            **kwargs:
        """
        super(PentaLinearScore, self).__init__()
        self.wemd_size = wemd_size
        self.tagset_size = tagset_size
        self.temd_size = temd_size
        self.rank = rank
        self.std = std

        self.tag_emd = nn.Parameter(torch.Tensor(self.tagset_size, self.temd_size))
        self.S = nn.Parameter(torch.Tensor(self.wemd_size, self.rank))
        self.T = nn.Parameter(torch.Tensor(self.wemd_size, self.rank))
        self.U = nn.Parameter(torch.Tensor(self.wemd_size, self.rank))
        self.V = nn.Parameter(torch.Tensor(self.temd_size, self.rank))
        self.W = nn.Parameter(torch.Tensor(self.temd_size, self.rank))
        self.first_wemb = nn.Parameter(torch.Tensor(1, self.wemd_size))
        self.last_wemb = nn.Parameter(torch.Tensor(1, self.wemd_size))
        # self.rand_init()

    def rand_init(self):
        '''random initialization
        '''

        utils.init_trans(self.tag_emd)
        utils.init_embedding(self.first_wemb)
        utils.init_embedding(self.last_wemb)
        utils.init_tensor(self.S, self.std)
        utils.init_tensor(self.T, self.std)
        utils.init_tensor(self.U, self.std)
        utils.init_tensor(self.V, self.std)
        utils.init_tensor(self.W, self.std)
        # std = 1.0
        # nn.init.xavier_normal_(self.tag_emd)
        # nn.init.xavier_normal_(self.U, gain=std)
        # nn.init.xavier_normal_(self.V, gain=std)
        # nn.init.xavier_normal_(self.W, gain=std)

    def forward(self, word_emd):
        """
        Args:

        Returns: Tensor
            the energy tensor with shape = [batch, num_label, length, length]
        """
        assert word_emd.size(2) == self.wemd_size, 'batch sizes of encoder and decoder are requires to be equal.'
        first_words = word_emd[:word_emd.size(0) - 1, :, :].clone()
        first_emds = self.first_wemb.view(1, 1, self.wemd_size).expand(1, word_emd.size(1), word_emd.size(2))
        first_word_emds = torch.cat([first_emds, first_words], 0)

        last_words = word_emd[1: word_emd.size(0), :, :].clone()
        last_emds = self.last_wemb.view(1, 1, self.wemd_size).expand(1, word_emd.size(1), word_emd.size(2))
        last_word_emds = torch.cat([last_words, last_emds], 0)
        # print(f'{word_emd1.size()}, {last_word.size()}, {word_emd.size()}')
        # (n x m x d) * (d x k) -> (n x m x k)
        g0 = torch.matmul(first_word_emds, self.S)
        g1 = torch.matmul(word_emd, self.U)
        g2 = torch.matmul(last_word_emds, self.T)
        g3 = torch.matmul(self.tag_emd, self.V)
        g4 = torch.matmul(self.tag_emd, self.W)
        # print(f'{g0.size}: {g1.size()}')
        temp01 = torch.einsum('nak, nak->nak', [g0, g1])
        temp012 = torch.einsum('nak, nak->nak', [temp01, g2])
        temp0123 = torch.einsum('nak, bk->nabk', [temp012, g3])
        score = torch.einsum('nabk, ck->nabc', [temp0123, g4])

        return score


class ConcatScore(nn.Module):
    """
    Outer product version of trilinear function.

    Trilinear attention layer.
    """

    def __init__(self, wemd_size, tagset_size, temd_size=20, rank=256, std=0.1545, method='Trilinear',**kwargs):
        """
        Args:
            input_size_encoder: int
                the dimension of the encoder input.
            input_size_decoder: int
                the dimension of the decoder input.
            num_labels: int
                the number of labels of the crf layer
            biaffine: bool
                if apply bi-affine parameter.
            **kwargs:
        """
        super(ConcatScore, self).__init__()
        print(method)
        self.wemd_size = wemd_size
        self.tagset_size = tagset_size
        self.temd_size = temd_size
        self.std = std
        self.method = method
        self.rank = rank

        self.tag_emd = nn.Parameter(torch.Tensor(self.tagset_size, self.temd_size))

        if 'Trilinear' == self.method:
            self.linear_layer = nn.Linear(self.wemd_size + self.temd_size * 2,
                                          self.rank)
        elif 'Qualinear' == self.method:
            self.linear_layer = nn.Linear(self.wemd_size * 2 + self.temd_size * 2,
                                          self.rank)
            self.first_wemb = nn.Parameter(torch.Tensor(1, self.wemd_size))
        self.vector = nn.Parameter(torch.Tensor(1, self.rank))

        self.tanH = nn.Tanh()

        # self.rand_init()

    def rand_init(self):
        '''random initialization
        '''

        utils.init_trans(self.tag_emd)
        if 'Qualinear' == self.method:
            utils.init_embedding(self.first_wemb)
        utils.init_embedding(self.vector)
        utils.init_linear(self.linear_layer)
        # std = 1.0
        # nn.init.xavier_normal_(self.tag_emd)
        # nn.init.xavier_normal_(self.U, gain=std)
        # nn.init.xavier_normal_(self.V, gain=std)
        # nn.init.xavier_normal_(self.W, gain=std)

    def forward(self, word_emd):
        """
        Args:

        Returns: Tensor
            the energy tensor with shape = [batch, num_label, length, length]
        """
        assert word_emd.size(2) == self.wemd_size, 'batch sizes of encoder and decoder are requires to be equal.'
        b, s = word_emd.size(0), word_emd.size(1)
        if 'Qualinear' == self.method:
            prev_word_emd = word_emd[:word_emd.size(0) - 1, :, :].clone()
            first_word = self.first_wemb.view(1, 1, self.wemd_size).expand(1, word_emd.size(1), word_emd.size(2))
            prev_word_emd = torch.cat([first_word, prev_word_emd], 0)
            word_emd = torch.cat([prev_word_emd, word_emd], 2)
        # print(f'{word_emd1.size()}, {last_word.size()}, {word_emd.size()}')
        # (n x m x d) * (d x k) -> (n x m x k)
        word_temp = word_emd.unsqueeze(2).repeat(1, 1, self.tagset_size, 1).unsqueeze(2).repeat(1, 1, self.tagset_size, 1, 1)
        tag_temp1 = self.tag_emd.unsqueeze(0).repeat(self.tagset_size, 1, 1)
        tag_temp2 = self.tag_emd.unsqueeze(1).repeat(1, self.tagset_size, 1)
        tag_temp = torch.cat([tag_temp1, tag_temp2], 2)
        tag_temp = tag_temp.unsqueeze(0).repeat(s, 1, 1, 1).unsqueeze(0).repeat(b, 1, 1, 1, 1)
        whole_rep = torch.cat([word_temp, tag_temp], 4)

        linear_outs = self.linear_layer(whole_rep)
        tanh_outs = self.tanH(linear_outs)
        score = torch.matmul(tanh_outs, self.vector.view(-1, 1))

        return score


class CRF_TRI(nn.Module):
    '''Conditional Random Field (CRF) layer with trilinear, Which is based on CRF_S.

    args:
        hidden_dim: input_size
        tagset_size: tag_set_size
        if_biase: whether use bias in linear layer
    '''

    def __init__(self, hidden_dim, tagset_size, dim=200, if_bias=True):
        super(CRF_TRI, self).__init__()
        self.tagset_size = tagset_size

        self.trans = nn.Parameter(torch.Tensor(self.tagset_size, dim))
        self.U = nn.Parameter(torch.Tensor(hidden_dim, 200))
        self.V = nn.Parameter(torch.Tensor(dim, 200))
        self.W = nn.Parameter(torch.Tensor(dim, 200))

    def rand_init(self):
        '''random initialization
        '''

        # utils.init_linear(self.hidden2tag)
        utils.init_trans(self.trans)
        utils.init_tensor(self.U)
        utils.init_tensor(self.V)
        utils.init_tensor(self.W)
        # self.transitions.data.zero_()

    def rand_test(self):
        self.transitions.data.zero_()

    def forward(self, hidden):
        '''s
        args:
            feats (bat_size, seq_len, hidden_dim): input score from lstm layer
        return:
            output from crf((bat_size * seq_len), tag_size, tag_size, tag_size)
        '''

        dim = self.U.size()[1]
        ins_num = hidden.size(0)

        first_g = torch.mm(hidden, self.U)
        # first_g = torch.tensordot(hidden, self.U, 1)
        second_g = self.trans.mm(self.V)
        third_g = self.trans.mm(self.W)

        second_g = second_g.view(self.tagset_size, 1, dim).expand(self.tagset_size, self.tagset_size, dim)
        third_g = third_g.view(1, self.tagset_size, dim).expand(self.tagset_size, self.tagset_size, dim)
        temp1 = first_g.unsqueeze(1).repeat(1, self.tagset_size, 1)
        temp1 = temp1.unsqueeze(1).repeat(1, self.tagset_size, 1, 1)
        temp2 = second_g * third_g

        temp2 = temp2.unsqueeze(0).repeat(ins_num, 1, 1, 1)
        crf_scores = torch.sum(temp1 * temp2, -1)

        return crf_scores


class CRF_S_SEC(nn.Module):
    '''Second order Conditional Random Field (CRF) layer, Which is based on CRF_S.

    args:
        hidden_dim: input_size
        tagset_size: tag_set_size
        if_biase: whether use bias in linear layer
    '''

    def __init__(self, hidden_dim, tagset_size, if_bias=True):
        super(CRF_S_SEC, self).__init__()
        self.tagset_size = tagset_size
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size, bias=if_bias)
        self.transitions = nn.Parameter(torch.Tensor(self.tagset_size, self.tagset_size, self.tagset_size))

    def rand_init(self):
        '''random initialization
        '''

        utils.init_linear(self.hidden2tag)
        self.transitions.data.zero_()

    def rand_test(self):
        self.transitions.data.zero_()

    def forward(self, feats):
        '''s
        args:
            feats (bat_size, seq_len, hidden_dim): input score from lstm layer
        return:
            output from crf((bat_size * seq_len), tag_size, tag_size, tag_size)
        '''

        scores = self.hidden2tag(feats).view(-1, 1, 1, self.tagset_size)
        ins_num = scores.size(0)
        crf_scores = scores.expand(ins_num, self.tagset_size, self.tagset_size, self.tagset_size) + \
                     self.transitions.view(1, self.tagset_size, self.tagset_size, self.tagset_size) \
                         .expand(ins_num, self.tagset_size, self.tagset_size, self.tagset_size)

        return crf_scores


class CRFLoss_vb(nn.Module):
    """loss for viterbi decode

    .. math::
        \sum_{j=1}^n \log (\phi(\hat{y}_{j-1}, \hat{y}_j, \mathbf{z}_j)) - \log (\sum_{\mathbf{y}' \in \mathbf{Y}(\mathbf{Z})} \prod_{j=1}^n \phi(y'_{j-1}, y'_j, \mathbf{z}_j) )

    args:
        tagset_size: target_set_size
        start_tag: ind for <start>
        end_tag: ind for <pad>
        average_batch: whether average the loss among batch

    """

    def __init__(self, tagset_size, start_tag, end_tag, average_batch=True):
        super(CRFLoss_vb, self).__init__()
        self.tagset_size = tagset_size
        self.start_tag = start_tag
        self.end_tag = end_tag
        self.average_batch = average_batch

    def forward(self, scores, target, mask):
        """
        args:
            scores (seq_len, bat_size, target_size_from, target_size_to) : crf scores
            target (seq_len, bat_size, 1) : golden state
            mask (size seq_len, bat_size) : mask for padding
        return:
            loss
        """

        # calculate batch size and seq len
        # scores = torch.Tensor(3,1,21,21).fill_(1).cuda()
        # target = torch.LongTensor(3,1,1).fill_(1).cuda()
        # mask = torch.ByteTensor(3,1).fill_(1).cuda()
        # self.tagset_size = 21
        seq_len = scores.size(0)
        bat_size = scores.size(1)

        # calculate sentence score
        # sc = 0.0
        # # for b in range(bat_size):
        # start_tag = torch.tensor([self.start_tag]).cuda()
        # start = start_tag[:, None].repeat(1, bat_size)
        # end = torch.tensor([self.end_tag]).cuda()
        # start_label = torch.cat([start, target.view(seq_len, bat_size)[:-1, :]], 0).view(seq_len*bat_size) % self.tagset_size
        # end_label = target.view(seq_len, bat_size).view(seq_len*bat_size) % self.tagset_size
        # score_ = scores.view(seq_len*bat_size, self.tagset_size, self.tagset_size)
        # ss = score_[range(seq_len*bat_size), start_label, end_label]
        # ss_energy = ss.view(seq_len, bat_size).masked_select(mask).sum()

        # print(seq_len, bat_size)
        # print(scores.size(), target.size())
        # print(target)
        tg_energy = torch.gather(scores.view(seq_len, bat_size, -1), 2, target).view(seq_len,
                                                                                     bat_size)  # seq_len * bat_size
        tg_energy = tg_energy.masked_select(mask).sum()

        # calculate forward partition score

        # build iter
        seq_iter = enumerate(scores)
        # the first score should start with <start>
        _, inivalues = seq_iter.__next__()  # bat_size * from_target_size * to_target_size
        # only need start from start_tag
        partition = inivalues[:, self.start_tag, :].clone()  # bat_size * to_target_size
        # iter over last scores
        for idx, cur_values in seq_iter:
            # previous to_target is current from_target
            # partition: previous results log(exp(from_target)), #(batch_size * from_target)
            # cur_values: bat_size * from_target * to_target
            cur_values = cur_values + partition.contiguous().view(bat_size, self.tagset_size, 1).expand(bat_size,
                                                                                                        self.tagset_size,
                                                                                                        self.tagset_size)

            cur_partition = utils.log_sum_exp(cur_values, self.tagset_size)
            # (bat_size * from_target * to_target) -> (bat_size * to_target)
            # partition = utils.switch(partition, cur_partition, mask[idx].view(bat_size, 1).expand(bat_size, self.tagset_size)).view(bat_size, -1)
            mask_idx = mask[idx, :].view(bat_size, 1).expand(bat_size, self.tagset_size)
            partition.masked_scatter_(mask_idx,
                                      cur_partition.masked_select(mask_idx))  # 0 for partition, 1 for cur_partition

        # only need end at end_tag
        partition = partition[:, self.end_tag].sum()
        # average = mask.sum()

        # average_batch
        # if self.average_batch:
        #     loss = (partition - tg_energy) / bat_size
        # else:
        loss = (partition - tg_energy) / bat_size

        return loss


class CRF_SECLoss_vb(nn.Module):
    """second order crf loss for viterbi decode

    .. math::


    args:
        tagset_size: target_set_size
        start_tag: ind for <start>
        end_tag: ind for <pad>
        average_batch: whether average the loss among batch

    """

    def __init__(self, tagset_size, start_tag, end_tag, average_batch=True):
        super(CRF_SECLoss_vb, self).__init__()
        self.tagset_size = tagset_size
        self.start_tag = start_tag
        self.end_tag = end_tag
        self.average_batch = average_batch

    def forward(self, scores, target, mask):
        """
        args:
            scores (seq_len, bat_size, target_size_from, target_size_to) : crf scores
            target (seq_len, bat_size, 1) : golden state
            mask (size seq_len, bat_size) : mask for padding
        return:
            loss
        """

        # calculate batch size and seq len
        seq_len = scores.size(0)
        bat_size = scores.size(1)

        # calculate sentence score
        tg_energy = torch.gather(scores.view(seq_len, bat_size, -1), 2, target).view(seq_len,
                                                                                     bat_size)  # seq_len * bat_size
        tg_energy = tg_energy.masked_select(mask).sum()

        # calculate forward partition score

        # build iter
        seq_iter = enumerate(scores)
        # the first score should start with (<start>, <start>)
        _, inivalues = seq_iter.__next__()  # bat_size * from_target_size * mid_target_size
        _, secvalues = seq_iter.__next__()
        # only need start from start_tag
        partition1 = inivalues[:, self.start_tag, self.start_tag, :].clone()  # bat_size * to_target_size
        partition2 = secvalues[:, self.start_tag, :, :].clone()
        partition_ = partition1.contiguous().view(bat_size, self.tagset_size, 1).expand(bat_size, self.tagset_size,
                                                                                        self.tagset_size)

        partition = partition_ + partition2.contiguous().view(bat_size, self.tagset_size, self.tagset_size)

        # iter over last scores
        for idx, cur_values in seq_iter:
            # previous to_target is current from_target
            # partition: previous results log(exp(from_target)), #(batch_size * from_target)
            # cur_values: bat_size * from_target * to_target
            cur_values = cur_values + partition.contiguous().view(bat_size, self.tagset_size, self.tagset_size,
                                                                  1).expand(bat_size,
                                                                            self.tagset_size,
                                                                            self.tagset_size,
                                                                            self.tagset_size)
            # TODO 每个partion添加25行0，形成26 X 26
            cur_partition = utils.log_sum_exp_sec(cur_values, self.tagset_size)
            # (bat_size * from_target * to_target) -> (bat_size * to_target)
            # partition = utils.switch(partition, cur_partition, mask[idx].view(bat_size, 1).expand(bat_size, self.tagset_size)).view(bat_size, -1)
            # mask_idx = mask[idx, :].view(bat_size, 1).expand(bat_size, self.tagset_size)
            mask_idx = mask[idx, :].view(bat_size, 1, 1).expand(bat_size, self.tagset_size, self.tagset_size)
            partition.masked_scatter_(mask_idx,
                                      cur_partition.masked_select(mask_idx))  # 0 for partition, 1 for cur_partition

        # only need end at end_tag
        partition = partition[:, self.end_tag, self.end_tag].sum()
        # average = mask.sum()

        # average_batch
        if self.average_batch:
            loss = (partition - tg_energy) / bat_size
        else:
            loss = (partition - tg_energy)

        return loss


class CRFDecode_vb():
    """Batch-mode viterbi decode

    args:
        tagset_size: target_set_size
        start_tag: ind for <start>
        end_tag: ind for <pad>
        average_batch: whether average the loss among batch

    """

    def __init__(self, tagset_size, start_tag, end_tag, average_batch=True):
        self.tagset_size = tagset_size
        self.start_tag = start_tag
        self.end_tag = end_tag
        self.average_batch = average_batch

    def decode(self, scores, mask):
        """Find the optimal path with viterbe decode

        args:
            scores (size seq_len, bat_size, target_size_from, target_size_to) : crf scores
            mask (seq_len, bat_size) : mask for padding
        return:
            decoded sequence (size seq_len, bat_size)
        """
        # calculate batch size and seq len

        seq_len = scores.size(0)
        bat_size = scores.size(1)

        mask = ~mask
        decode_idx = torch.LongTensor(seq_len - 1, bat_size)

        # calculate forward score and checkpoint

        # build iter
        seq_iter = enumerate(scores)
        # the first score should start with <start>
        _, inivalues = seq_iter.__next__()  # bat_size * from_target_size * to_target_size
        # only need start from start_tag
        forscores = inivalues[:, self.start_tag, :]  # bat_size * to_target_size
        back_points = list()
        # iter over last scores
        for idx, cur_values in seq_iter:
            # previous to_target is current from_target
            # partition: previous results log(exp(from_target)), #(batch_size * from_target)
            # cur_values: bat_size * from_target * to_target
            cur_values = cur_values + forscores.contiguous().view(bat_size, self.tagset_size, 1).expand(bat_size,
                                                                                                        self.tagset_size,
                                                                                                        self.tagset_size)

            forscores, cur_bp = torch.max(cur_values, 1)
            cur_bp.masked_fill_(mask[idx].view(bat_size, 1).expand(bat_size, self.tagset_size), self.end_tag)
            back_points.append(cur_bp)

        pointer = back_points[-1][:, self.end_tag]
        decode_idx[-1] = pointer
        for idx in range(len(back_points) - 2, -1, -1):
            pointer = torch.gather(back_points[idx], 1, pointer.contiguous().view(bat_size, 1))
            decode_idx[idx] = pointer.view(-1, )
        return decode_idx


class CRFSECDecode_vb():
    """ Sec order crf Batch-mode viterbi decode

    args:
        tagset_size: target_set_size
        start_tag: ind for <start>
        end_tag: ind for <pad>
        average_batch: whether average the loss among batch

    """

    def __init__(self, tagset_size, start_tag, end_tag, average_batch=True):
        self.tagset_size = tagset_size
        self.start_tag = start_tag
        self.end_tag = end_tag
        self.average_batch = average_batch

    def decode(self, scores, mask):
        """Find the optimal path with viterbe decode

        args:
            scores (size seq_len, bat_size, target_size_from, target_size_to) : crf scores
            mask (seq_len, bat_size) : mask for padding
        return:
            decoded sequence (size seq_len, bat_size)
        """
        # calculate batch size and seq len

        seq_len = scores.size(0)
        bat_size = scores.size(1)

        mask = 1 - mask
        decode_idx = torch.LongTensor(seq_len - 2, bat_size)  # -1 for the last is '<eof>'

        # calculate forward score and checkpoint

        # build iter
        seq_iter = enumerate(scores)
        '''
        _, inivalues = seq_iter.__next__()  # bat_size * from_target_size * mid_target_size
        _, secvalues = seq_iter.__next__()
        # only need start from start_tag
        partition1 = inivalues[:, self.start_tag, self.start_tag, :].clone()  # bat_size * to_target_size
        partition2 = secvalues[:, self.start_tag, :, :]
        partition_ = partition1.contiguous().view(bat_size, self.tagset_size, 1).expand(bat_size, self.tagset_size,
                                                                                        self.tagset_size)
        partition = partition_ + partition2.contiguous().view(bat_size, self.tagset_size, self.tagset_size)
        '''
        # the first score should start with <start>
        _, inivalues = seq_iter.__next__()  # bat_size * from_target_size * to_target_size
        _, secvalues = seq_iter.__next__()  # ~
        # only need start from (start_tag, start_tag)
        forscores1 = inivalues[:, self.start_tag, self.start_tag, :]  # bat_size * to_target_size
        forscores2 = secvalues[:, self.start_tag, :, :]

        forscores_ = forscores1.contiguous().view(bat_size, self.tagset_size, 1).expand(bat_size, self.tagset_size,
                                                                                        self.tagset_size)
        forscores = forscores_ + forscores2.contiguous().view(bat_size, self.tagset_size, self.tagset_size)

        back_points = list()
        # iter over last scores
        for idx, cur_values in seq_iter:
            # previous to_target is current from_target
            # partition: previous results log(exp(from_target)), #(batch_size * from_target)
            # cur_values: bat_size * from_target * to_target
            cur_values = cur_values + forscores.contiguous().view(bat_size, self.tagset_size, self.tagset_size,
                                                                  1).expand(bat_size,
                                                                            self.tagset_size,
                                                                            self.tagset_size,
                                                                            self.tagset_size)

            forscores, cur_bp = torch.max(cur_values, 1)  # B * 1 * M * M
            # forscores, cur_bp = torch.max(cur_values, 1)
            # TODO: 是 idx - 1 吗？ 还需要确认下
            cur_bp.masked_fill_(mask[idx].view(bat_size, 1, 1).expand(bat_size, self.tagset_size, self.tagset_size),
                                self.end_tag)
            back_points.append(cur_bp)

        pointer1 = back_points[-1][:, self.end_tag, self.end_tag]
        decode_idx[-1] = pointer1

        pointer_ = back_points[-2][:, :, self.end_tag].view(bat_size, self.tagset_size)
        pointer1 = pointer1.contiguous().view(bat_size, 1)
        pointer2 = torch.gather(pointer_, 1, pointer1)
        decode_idx[-2] = pointer2.view(-1, )

        pointer1 = pointer1.view(bat_size, 1, 1).expand(bat_size, self.tagset_size, 1)
        pointer2 = pointer2.contiguous().view(bat_size, 1)

        for idx in range(len(back_points) - 3, -1, -1):
            pointer_ = torch.gather(back_points[idx], 2, pointer1).contiguous().view(bat_size, self.tagset_size)
            pointer = torch.gather(pointer_, 1, pointer2)
            decode_idx[idx] = pointer.view(-1, )

            pointer1 = pointer2.view(bat_size, 1, 1).expand(bat_size, self.tagset_size, 1)
            pointer2 = pointer.contiguous().view(bat_size, 1)
        return decode_idx
