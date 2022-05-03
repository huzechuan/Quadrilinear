import os
import re
from abc import abstractmethod
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import List, Union, Dict
import pickle
import gensim
import numpy as np
import torch
import torchvision as torchvision
# from bpemb import BPEmb
# from deprecated import deprecated
from torch.nn import ParameterList, Parameter
from data_process.data import BertFormat
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    BertTokenizer,
    BertModel,
    BertConfig,
    RobertaTokenizer,
    RobertaModel,
    TransfoXLTokenizer,
    TransfoXLModel,
    OpenAIGPTModel,
    OpenAIGPTTokenizer,
    GPT2Model,
    GPT2Tokenizer,
    XLNetTokenizer,
    XLMTokenizer,
    XLNetModel,
    XLMModel,
    PreTrainedTokenizer,
    PreTrainedModel,
    AutoTokenizer,
    AutoConfig,
    AutoModel,
    XLMRobertaConfig,
    XLMRobertaModel,
    XLMRobertaTokenizer
)


from data_process.data import Corpus

from data_process.file_utils import cached_path

import models.utils as utils

root = '.TL/'

class Embeddings(torch.nn.Module):
    """Abstract base class for all embeddings. Every new type of embedding must implement these methods."""

    @property
    @abstractmethod
    def embedding_length(self) -> int:
        """Returns the length of the embedding vector."""
        pass

    @property
    @abstractmethod
    def embedding_type(self) -> str:
        pass

    def embed(self, sentences):
        """Add embeddings to all words in a list of sentences. If embeddings are already added, updates only if embeddings
        are non-static."""

        # if only one sentence is passed, convert to list of sentence
        pass

    @abstractmethod
    def _add_embeddings_internal(self, sentences):
        """Private method for adding embeddings to all words in a list of sentences."""
        pass


class TokenEmbeddings(Embeddings):
    """Abstract base class for all token-level embeddings. Ever new type of word embedding must implement these methods."""

    @property
    @abstractmethod
    def embedding_length(self) -> int:
        """Returns the length of the embedding vector."""
        pass

    @property
    def embedding_type(self) -> str:
        return "word-level"


class StackedEmbeddings(TokenEmbeddings):
    """A stack of embeddings, used if you need to combine several different embedding types."""

    def __init__(self, embeddings: List[TokenEmbeddings]):
        """The constructor takes a list of embeddings to be combined."""
        super().__init__()

        self.embeddings = embeddings

        # IMPORTANT: add embeddings as torch modules
        for i, embedding in enumerate(embeddings):
            self.add_module("list_embedding_{}".format(i), embedding)

        self.name: str = "Stack"
        self.static_embeddings: bool = False

        self.__embedding_type: str = embeddings[0].embedding_type

        self.__embedding_length: int = 0
        for embedding in embeddings:
            self.__embedding_length += embedding.embedding_length

    def embed(
        self, sentences
    ):
        # if only one sentence is passed, convert to list of sentence
        embed = list()
        for embedding in self.embeddings:
            embed.append(embedding.embed_sentences(sentences))
        if len(embed) > 1:
            embed = torch.cat(embed, 2)
        else:
            embed = embed[0]
        return embed

    @property
    def embedding_type(self) -> str:
        return self.__embedding_type

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences):

        for embedding in self.embeddings:
            embedding._add_embeddings_internal(sentences)

        return sentences

    def __str__(self):
        return f'StackedEmbeddings [{",".join([str(e) for e in self.embeddings])}]'


class WordEmbeddings(TokenEmbeddings):
    """Standard Fine Tune word embeddings, such as GloVe or FastText."""

    def __init__(self, embeddings: str, all_tokens: list, field: str = None, if_cased: bool = False):
        """
        Initializes classic word embeddings. Constructor downloads required files if not there.
        :param embeddings: one of: 'glove', 'extvec', 'crawl' or two-letter language code or custom
        If you want to use a custom embedding file, just pass the path to the embeddings as embeddings variable.
        """
        super().__init__()
        embed_name = embeddings
        old_base_path = (
            "https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings/"
        )
        base_path = (
            "https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.3/"
        )
        embeddings_path_v4 = (
            "https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.4/"
        )
        embeddings_path_v4_1 = "https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/embeddings-v0.4.1/"

        cache_dir = Path("embeddings")

        # GLOVE embeddings
        if embeddings.lower() == "glove" or embeddings.lower() == "en-glove":
            cached_path(f"{old_base_path}glove.gensim.vectors.npy", cache_dir=cache_dir)
            embeddings = cached_path(
                f"{old_base_path}glove.gensim", cache_dir=cache_dir
            )

        # TURIAN embeddings
        elif embeddings.lower() == "turian" or embeddings.lower() == "en-turian":
            cached_path(
                f"{embeddings_path_v4_1}turian.vectors.npy", cache_dir=cache_dir
            )
            embeddings = cached_path(
                f"{embeddings_path_v4_1}turian", cache_dir=cache_dir
            )

        # KOMNINOS embeddings
        elif embeddings.lower() == "extvec" or embeddings.lower() == "en-extvec":
            cached_path(
                f"{old_base_path}extvec.gensim.vectors.npy", cache_dir=cache_dir
            )
            embeddings = cached_path(
                f"{old_base_path}extvec.gensim", cache_dir=cache_dir
            )

        # FT-CRAWL embeddings
        elif embeddings.lower() == "crawl" or embeddings.lower() == "en-crawl":
            cached_path(
                f"{base_path}en-fasttext-crawl-300d-1M.vectors.npy", cache_dir=cache_dir
            )
            embeddings = cached_path(
                f"{base_path}en-fasttext-crawl-300d-1M", cache_dir=cache_dir
            )

        # FT-CRAWL embeddings
        elif (
            embeddings.lower() == "news"
            or embeddings.lower() == "en-news"
            or embeddings.lower() == "en"
        ):
            cached_path(
                f"{base_path}en-fasttext-news-300d-1M.vectors.npy", cache_dir=cache_dir
            )
            embeddings = cached_path(
                f"{base_path}en-fasttext-news-300d-1M", cache_dir=cache_dir
            )

        # twitter embeddings
        elif embeddings.lower() == "twitter" or embeddings.lower() == "en-twitter":
            cached_path(
                f"{old_base_path}twitter.gensim.vectors.npy", cache_dir=cache_dir
            )
            embeddings = cached_path(
                f"{old_base_path}twitter.gensim", cache_dir=cache_dir
            )

        # two-letter language code wiki embeddings
        elif len(embeddings.lower()) == 2:
            cached_path(
                f"{embeddings_path_v4}{embeddings}-wiki-fasttext-300d-1M.vectors.npy",
                cache_dir=cache_dir,
            )
            embeddings = cached_path(
                f"{embeddings_path_v4}{embeddings}-wiki-fasttext-300d-1M",
                cache_dir=cache_dir,
            )

        # two-letter language code wiki embeddings
        elif len(embeddings.lower()) == 7 and embeddings.endswith("-wiki"):
            cached_path(
                f"{embeddings_path_v4}{embeddings[:2]}-wiki-fasttext-300d-1M.vectors.npy",
                cache_dir=cache_dir,
            )
            embeddings = cached_path(
                f"{embeddings_path_v4}{embeddings[:2]}-wiki-fasttext-300d-1M",
                cache_dir=cache_dir,
            )

        # two-letter language code crawl embeddings
        elif len(embeddings.lower()) == 8 and embeddings.endswith("-crawl"):
            cached_path(
                f"{embeddings_path_v4}{embeddings[:2]}-crawl-fasttext-300d-1M.vectors.npy",
                cache_dir=cache_dir,
            )
            embeddings = cached_path(
                f"{embeddings_path_v4}{embeddings[:2]}-crawl-fasttext-300d-1M",
                cache_dir=cache_dir,
            )
        elif embeddings.lower().startswith('conll_'):
            embeddings = Path(root) / cache_dir / f'{embeddings.lower()}.txt'
        elif embeddings.lower().endswith('cc.el'):
            embeddings = Path(root) / cache_dir / f'{embeddings.lower()}.300.txt'
        elif not Path(embeddings).exists():
            raise ValueError(
                f'The given embeddings "{embeddings}" is not available or is not a valid path.'
            )

        self.static_embeddings = False

        if str(embeddings).endswith(".bin"):
            precomputed_word_embeddings = gensim.models.KeyedVectors.load_word2vec_format(
                str(embeddings), binary=True
            )
        elif str(embeddings).endswith('.txt'):
            precomputed_word_embeddings = gensim.models.KeyedVectors.load_word2vec_format(
                str(embeddings), binary=False
            )
        else:
            precomputed_word_embeddings = gensim.models.KeyedVectors.load(
                str(embeddings)
            )

        self.field = field

        self.name = f'Word: {embed_name} if_cased: {if_cased}'
        self.__embedding_length: int = precomputed_word_embeddings.vector_size
        self.if_cased = if_cased
        self.get = self.get_idx_cased if if_cased else self.get_idx

        if if_cased:  # Usually NER use cased
            train_set = set([token for token in all_tokens[0]])  # | set([token.lower() for token in all_tokens[0]])
            full_set = set([token for token in all_tokens[1]])  # | set([token.lower() for token in all_tokens[1]])
        else:
            train_set = set([token.lower() for token in all_tokens[0]])
            full_set = set([token.lower() for token in all_tokens[1]])

        vocab = precomputed_word_embeddings.vocab
        self.vocab = {}
        self.vocab['unk'] = len(self.vocab)

        emb_vecs = precomputed_word_embeddings.vectors
        if 'unk' in vocab.keys():
            embeddings_tmp = [torch.FloatTensor(emb_vecs[vocab['unk'].index]).unsqueeze(0)]
        else:
            embeddings_tmp = [utils.rand_emb(
                torch.FloatTensor(self.__embedding_length)
            ).unsqueeze(0)]

        in_train = True
        train_emb = 0
        train_rand = 0
        for token in full_set:
            if token in vocab.keys():
                word_embedding = torch.FloatTensor(emb_vecs[vocab[token].index])
                train_emb += 1
            elif if_cased and (token.lower() in vocab.keys()):
                word_embedding = torch.FloatTensor(emb_vecs[vocab[token.lower()].index])
                train_emb += 1
            else:
                if token in train_set:
                    word_embedding = utils.rand_emb(
                        torch.FloatTensor(self.__embedding_length)
                    )
                else:
                    in_train = False
                    pass

                # word_embedding = word_embedding.view(word_embedding.size()[0]*word_embedding.size()[1])
            if in_train:
                if token not in self.vocab:
                    embeddings_tmp.append(word_embedding.unsqueeze(0))
                    self.vocab[token] = len(self.vocab)
                    train_rand += 1
            else:
                in_train = True
        for i in range(3):
            embeddings_tmp.append(utils.rand_emb(
                            torch.FloatTensor(self.__embedding_length)
                        ).unsqueeze(0))
        self.vocab['<start>'] = len(self.vocab)
        self.vocab['<end>'] = len(self.vocab)
        self.vocab['<eof>'] = len(self.vocab)
        embeddings_tmp = torch.cat(embeddings_tmp, 0)
        assert len(self.vocab) == embeddings_tmp.size()[0], "vocab_dic and embedding size not match!"
        self.word_embedding = torch.nn.Embedding(embeddings_tmp.shape[0], embeddings_tmp.shape[1])
        self.word_embedding.weight = torch.nn.Parameter(embeddings_tmp)

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def get_idx(self, token: str):
        return self.vocab.get(token.lower(), self.vocab['unk'])

    def get_idx_cased(self, token: str):
        return self.vocab.get(token, self.vocab.get(token.lower(), self.vocab['unk']))

    def encode_sentences(self, sentence):
        tokens = sentence.tokens
        lines = torch.LongTensor(list(map(self.get, tokens)))
        embed = sentence.set_word(lines.unsqueeze(0))
        return embed

    def embed_sentences(self, sentences):
        embeds = []
        for sentence in sentences:
            if sentence.word_id is None:
                embeds.append(self.encode_sentences(sentence))
            else:
                embeds.append(sentence.word_id.cuda())

        embeds = torch.cat(embeds, 0)
        embeddings = self.word_embedding(embeds)

        return embeddings.transpose(0, 1)

    def __str__(self):
        return self.name

class CharacterEmbeddings(TokenEmbeddings):
    """Character embeddings of words, as proposed in Lample et al., 2016."""

    def __init__(
        self,
        vocab: dict = None,
        char_embedding_dim: int = 25,
        hidden_size_char: int = 25,
    ):
        """Uses the default character dictionary if none provided."""

        super().__init__()
        self.name = "Char"
        self.static_embeddings = False

        # use list of common characters if none provided
        self.char_dictionary = {'<u>': 0}
        for word in vocab:
            for c in word:
                if c not in self.char_dictionary:
                    self.char_dictionary[c] = len(self.char_dictionary)

        self.char_dictionary[' '] = len(self.char_dictionary)  # concat for char
        self.char_dictionary['\n'] = len(self.char_dictionary)  # eof for char

        self.char_embedding_dim: int = char_embedding_dim
        self.hidden_size_char: int = hidden_size_char
        self.char_embedding = torch.nn.Embedding(
            len(self.char_dictionary), self.char_embedding_dim
        )
        self.char_rnn = torch.nn.LSTM(
            self.char_embedding_dim,
            self.hidden_size_char,
            num_layers=1,
            bidirectional=True,
        )

        self.__embedding_length = self.char_embedding_dim * 2
        self.pad_word = utils.rand_emb(torch.FloatTensor(self.__embedding_length)).unsqueeze(0).unsqueeze(0)

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def embed_sentences(self, sentences):
        sentences = [sent.tokens for sent in sentences]
        lens = [[len(token) for token in sent] for sent in sentences]
        max_len = max(list(map(lambda s: max(s), lens)))

        lines = list(map(lambda t: list(map(lambda m: [[0] * (len(m) - 1) + [1] + [0] * (max_len - len(m)),
                                                       list(map(lambda c: self.char_dictionary.get(c, 0), m)) + [0] * (
                                                                   max_len - len(m))], t)), sentences))
        ten = torch.LongTensor(lines)
        batch = ten[:, :, 1].size(0)
        seq = ten[:, :, 1].size(1)

        embeds_idx = ten[:, :, 1].view(-1, max_len).transpose(0, 1).cuda()
        mask = ten[:, :, 0].bool().view(-1, max_len).unsqueeze(2).repeat(1, 1, self.__embedding_length).cuda()

        embeddings = self.char_embedding(embeds_idx)
        lstm_out, hidden = self.char_rnn(embeddings)

        outs = lstm_out.transpose(0, 1)

        outs = outs.masked_select(mask).view(lstm_out.size(1), -1).view(batch, seq, self.__embedding_length)

        return outs.transpose(0, 1)
    # def encode_sentence(self, sentence):
    #     length = sentence.length
    #     max_len = max([len(token) for token in sentence.tokens[:length]])
    #     lines = list(map(lambda m: [[0] * (len(m) - 1) + [1] + [0] * (max_len-len(m)), list(map(lambda c: self.char_dictionary.get(c, 0), m)) + [0] * (max_len - len(m))], sentence.tokens[:length]))
    #     lines = torch.LongTensor(lines)
    #     char_id = lines[:, 1, :].transpose(0, 1)
    #     char_mask = lines[:, 0, :].bool().transpose(0, 1).view(max_len, length, 1)
    #     char_id, char_mask = sentence.set_char(char_id, char_mask)
    #     return char_id, char_mask
    #
    # def embed_sentences(self, sentences):
    #     embeddings = []
    #     for sentence in sentences:
    #         length = sentence.length
    #         if sentence.char_id is None:
    #             char_ids, char_masks = self.encode_sentence(sentence)
    #         else:
    #             char_ids, char_masks = sentence.get_char()
    #
    #         char_masks = char_masks.expand(-1, length, self.__embedding_length)
    #         char_embeds = self.char_embedding(char_ids)
    #         lstm_out, hidden = self.char_rnn(char_embeds)
    #
    #         outs = lstm_out.masked_select(char_masks).view(length, 1, self.__embedding_length)
    #         embedding_tmp = [outs] + [self.pad_word.cuda()] * (len(sentence.tokens) - length)
    #         embeddings.append(torch.cat(embedding_tmp, 0))
    #
    #     char_embeddings = torch.cat(embeddings, 1)
    #
    #     return char_embeddings

    def __str__(self):
        return self.name


class BertEmbeddings(TokenEmbeddings):
    def __init__(
        self,
        bert_model_or_path: str = "bert-base-uncased",
        layers: str = "-1,-2,-3,-4",
        # pooling_operation: str = "first",
        # use_scalar_mix: bool = False,
    ):
        """
        Bidirectional transformer embeddings of words, as proposed in Devlin et al., 2018.
        :param bert_model_or_path: name of BERT model ('') or directory path containing custom model, configuration file
        and vocab file (names of three files should be - config.json, pytorch_model.bin/model.chkpt, vocab.txt)
        :param layers: string indicating which layers to take for embedding
        :param pooling_operation: how to get from token piece embeddings to token embedding. Either pool them and take
        the average ('mean') or use first word piece embedding as token embedding ('first)
        """
        super().__init__()

        self.tokenizer = BertTokenizer.from_pretrained(bert_model_or_path)
        self.model = BertModel.from_pretrained(
            pretrained_model_name_or_path=bert_model_or_path, output_hidden_states=True
        )
        self.layer_indexes = [int(x) for x in layers.split(",")]
        # self.pooling_operation = pooling_operation
        # self.use_scalar_mix = use_scalar_mix
        self.name = str(bert_model_or_path)
        self.static_embeddings = True
        self.__embedding_length: int = len(self.layer_indexes) * self.model.config.hidden_size
        self.pad_word = utils.rand_emb(torch.FloatTensor(self.__embedding_length)).unsqueeze(0)

    class BertInputFeatures(object):
        """Private helper class for holding BERT-formatted features"""

        def __init__(
                self,
                unique_id,
                tokens,
                input_ids,
                input_mask,
                input_type_ids,
                token_subtoken_count,
        ):
            self.unique_id = unique_id
            self.tokens = tokens
            self.input_ids = input_ids
            self.input_mask = input_mask
            self.input_type_ids = input_type_ids
            self.token_subtoken_count = token_subtoken_count


    def _convert_sentences_to_features(
            self, sentences, max_sequence_length: int
    ) -> [BertInputFeatures]:

        max_sequence_length = max_sequence_length + 2

        features: List[BertEmbeddings.BertInputFeatures] = []
        for (sentence_index, sentence) in enumerate(sentences):

            bert_tokenization: List[str] = []
            token_subtoken_count: Dict[int, int] = {}

            for token in sentence:
                subtokens = self.tokenizer.tokenize(token)
                bert_tokenization.extend(subtokens)
                token_subtoken_count[sentence.index(token)] = len(subtokens)

            if len(bert_tokenization) > max_sequence_length - 2:
                bert_tokenization = bert_tokenization[0: (max_sequence_length - 2)]

            tokens = []
            input_type_ids = []
            tokens.append("[CLS]")
            input_type_ids.append(0)
            for token in bert_tokenization:
                tokens.append(token)
                input_type_ids.append(0)
            tokens.append("[SEP]")
            input_type_ids.append(0)

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)
            # Zero-pad up to the sequence length.
            while len(input_ids) < max_sequence_length:
                input_ids.append(0)
                input_mask.append(0)
                input_type_ids.append(0)

            features.append(
                BertEmbeddings.BertInputFeatures(
                    unique_id=sentence_index,
                    tokens=tokens,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    input_type_ids=input_type_ids,
                    token_subtoken_count=token_subtoken_count,
                )
            )

        return features

    def _add_embeddings_internal(self, sents):
        """Add embeddings to all words in a list of sentences. If embeddings are already added,
        updates only if embeddings are non-static."""
        sentences = [sent.tokens[:sent.length] for sent in sents]
        # first, find longest sentence in batch
        longest_sentence_in_batch: int = len(
            max(
                [
                    self.tokenizer.tokenize(sentence.tokenize())
                    for sentence in sents
                ],
                key=len,
            )
        )

        # prepare id maps for BERT model
        features = self._convert_sentences_to_features(
            sentences, longest_sentence_in_batch
        )
        all_input_ids = torch.LongTensor([f.input_ids for f in features]).cuda()
        all_input_masks = torch.LongTensor([f.input_mask for f in features]).cuda()

        # put encoded batch through BERT model to get all hidden states of all encoder layers
        with torch.no_grad():
            self.model.cuda()
            self.model.eval()
            _, _, all_encoder_layers = self.model(
                all_input_ids, token_type_ids=None, attention_mask=all_input_masks
            )

        with torch.no_grad():

            for sentence_index, sentence in enumerate(sentences):

                feature = features[sentence_index]

                # get aggregated embeddings for each BERT-subtoken in sentence
                subtoken_embeddings = []
                for token_index, _ in enumerate(feature.tokens):
                    all_layers = []
                    for layer_index in self.layer_indexes:

                        layer_output = (
                            all_encoder_layers[int(layer_index)]
                                .detach()
                                .cpu()[sentence_index]
                        )
                        all_layers.append(layer_output[token_index])

                    subtoken_embeddings.append(torch.cat(all_layers))

                    # get the current sentence object
                token_idx = 0
                tokens_emb = []
                for token in sentence:
                    # add concatenated embedding to sentence
                    token_idx += 1
                    tokens_emb.append(subtoken_embeddings[token_idx].unsqueeze(0))
                    # token.set_embedding(self.name, subtoken_embeddings[token_idx])

                    token_idx += feature.token_subtoken_count[sentence.index(token)] - 1

                sents[sentence_index].set_bert(tokens_emb)

    def embed_sentences(self, sentences):
        if self.if_None(sentences):
            self._add_embeddings_internal(sentences)
        embedding_temp = []
        for sentence in sentences:
            bert_temp = sentence.bertEmbedding + [self.pad_word] * (sentence.total_len - sentence.length)
            embedding_temp.append(torch.cat(bert_temp, 0).unsqueeze(1))

        embeddings = torch.cat(embedding_temp, 1)
        return embeddings.cuda()

    def if_None(self, sentences):
        for sentence in sentences:
            if sentence.bertEmbedding is None:
                return True
        return False

    @property
    @abstractmethod
    def embedding_length(self) -> int:
        """Returns the length of the embedding vector."""
        # return (
        #     len(self.layer_indexes) * self.model.config.hidden_size
        #     if not self.use_scalar_mix
        #     else self.model.config.hidden_size
        # )
        return self.__embedding_length

    def __str__(self):
        return self.name


class XLMRobertaEmbeddings(TokenEmbeddings):
    def __init__(self,
                 model: str = "bert-base--multilingual-cased",
                 fine_tune: bool = False,
                 device: str = 'cpu'):
        super().__init__()
        self.device = device
        if 'BERT' in str(model) or 'bert' in str(model):
            print(f'use bert...{model}')
            self.tokenizer = BertTokenizer.from_pretrained(model)
            config = BertConfig.from_pretrained(model)
            self.config = config
            self.model = BertModel.from_pretrained(model, config=config)
        else:
            print(f'use XLMR...{model}')
            self.tokenizer = XLMRobertaTokenizer.from_pretrained(model)
            config = XLMRobertaConfig.from_pretrained(model)
            self.config = config
            # config.num_labels = 11
            # self.model = AutoModelForTokenClassification.from_pretrained(model, config=config)
            self.model = XLMRobertaModel.from_pretrained(model, config=config)
        self.fine_tune = fine_tune
        # model name
        self.name = str(model)
        self.static_embeddings = True
        self.__embedding_length: int = self.model.config.hidden_size
        self.pad_word = utils.rand_emb(torch.FloatTensor(self.__embedding_length)).unsqueeze(0)

        # when initializing, embeddings are in eval mode by default
        self.model.eval()
        self.model.to(device)

    @property
    # @abstractmethod
    def embedding_length(self) -> int:
        """Returns the length of the embedding vector."""

        length = self.model.config.hidden_size
        return length

    def _remove_special_token(self, sub):
        text = re.sub('^##', '', sub)  # BERT models
        return text

    def __first_subword(self, subword, sentence):
        # for mBERT

        first_subword_pos = []
        w_id = 1
        if self.name.startswith('bert') or 'BERT' in self.name or 'bert' in self.name:
            for s_id, word in enumerate(sentence):
                pieces = self.tokenizer.encode(word, add_special_tokens=False)
                sub_l = len(pieces)
                first_subword_pos.append(w_id)
                w_id += sub_l

        # for XLM-R
        else:
            for idx, sub_w in enumerate(subword):
                if sub_w.startswith('▁'):
                    first_subword_pos.append(idx)
        return first_subword_pos


    def _add_embeddings_internal(self, sents):
        sentences = [sent.tokens[:sent.length] for sent in sents]
        # print(sentences)
        # exit(0)
        ids = []
        masks = []
        mask_text = []
        subwords = []
        for sent in sentences:
            sentence = ' '.join(sent)
            input_id = self.tokenizer.encode(sentence)
            ids.append(input_id)
            subword = self.tokenizer.convert_ids_to_tokens(input_id)
            subwords.append(self.__first_subword(subword, sent))
        max_len = len(max(ids, key=len))
        # max_text_len = len(max(sentences, key=len))
        mask_t = torch.zeros(
            [len(ids), max_len],
            dtype=torch.bool,
            device=self.device,
            requires_grad=False
        )
        mask = torch.zeros(
            [len(ids), max_len],
            dtype=torch.long,
            device=self.device,
            requires_grad=False
        )
        input_ids = torch.zeros(
            [len(ids), max_len],
            dtype=torch.long,
            device=self.device,
            requires_grad=False
        )
        for idx, sent in enumerate(ids):
            length = len(sent)
            input_ids[idx][:length] = torch.tensor(sent, dtype=torch.long)
            mask[idx][:length] = torch.ones(length)
            # mask_t[idx][:length] = torch.tensor(subwords[idx], dtype=torch.bool)

        gradient_context = torch.enable_grad() if (self.fine_tune and self.training) else torch.no_grad()

        with torch.no_grad():
            self.model.eval()
            scores = self.model(input_ids, attention_mask=mask)[0]
            features = []
            sent_features = scores[:, 0, :]

            for sentence_idx, first_id in enumerate(subwords):
                select_index = torch.tensor(first_id, dtype=torch.long, device=self.device)
                # get states from all selected layers, aggregate with pooling operation
                sent_states = torch.index_select(scores[sentence_idx], 0, select_index)
                features.append(sent_states)
                sents[sentence_idx].set_bert(sent_states.cpu())
            # scores = pad_sequence(features, padding_value=0.0, batch_first=True)

        return None

    def embed_sentences(self, sentences):
        if self.if_None(sentences):
            self._add_embeddings_internal(sentences)
        embedding_temp = []
        for sentence in sentences:
            bert_temp = [sentence.bertEmbedding] + [self.pad_word] * (sentence.total_len - sentence.length)
            embedding_temp.append(torch.cat(bert_temp, 0))#.unsqueeze(1))
            # embedding_temp.append(sentence.bertEmbedding)
            # print(sentence.total_len, sentence.length)
            # print(sentence.bertEmbedding.size(), embedding_temp[-1].size(), sentence.total_len, sentence.length, sentence.tokens)

        # embeddings = torch.cat(embedding_temp, 1)
        embeddings = pad_sequence(embedding_temp, padding_value=0.0, batch_first=False)
        # print(embeddings.size())
        return embeddings.cuda()

    def if_None(self, sentences):
        for sentence in sentences:
            if sentence.bertEmbedding is None:
                return True
        return False