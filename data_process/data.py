from typing import List, Dict, Union, Callable

import torch
import logging

from collections import Counter
from collections import defaultdict

from torch.utils.data import Dataset, random_split
from torch.utils.data.dataset import ConcatDataset, Subset

class Corpus:
    def __init__(
        self,
        train: Dataset,
        dev: Dataset,
        test: Dataset,
        name: str = "corpus",
    ):
        self._train: Dataset = train
        self._dev: Dataset = dev
        self._test: Dataset = test
        self.name: str = name

    @property
    def train(self) -> Dataset:
        return self._train

    @property
    def dev(self) -> Dataset:
        return self._dev

    @property
    def test(self) -> Dataset:
        return self._test

    def _get_most_common_tokens(self, max_tokens, min_freq) -> List[str]:
        tokens_and_frequencies = Counter(self._get_all_tokens())
        tokens_and_frequencies = tokens_and_frequencies.most_common()

        tokens = []
        for token, freq in tokens_and_frequencies:
            if (min_freq != -1 and freq < min_freq) or (
                max_tokens != -1 and len(tokens) == max_tokens
            ):
                break
            tokens.append(token)
        return tokens

    def get_train_full_tokenset(self, max_tokens, min_freq) -> List[list]:

        train_set = self._get_most_common_tokens(max_tokens, min_freq)

        full_sents = self.get_all_sentences()
        full_tokens = [token for sublist in full_sents for token in sublist]

        tokens_and_frequencies = Counter(full_tokens)
        tokens_and_frequencies = tokens_and_frequencies.most_common()

        full_set = []
        for token, freq in tokens_and_frequencies:
            if (min_freq != -1 and freq < min_freq) or (
                    max_tokens != -1 and len(full_set) == max_tokens
            ):
                break
            full_set.append(token)

        t_fset = list()
        t_fset.append(train_set)
        t_fset.append(full_set)

        return t_fset

    def _get_all_tokens(self) -> List[str]:
        tokens = self.train.sentences
        tokens = [token for sublist in tokens for token in sublist]
        return tokens

    def __str__(self) -> str:
        return "Corpus: %d train + %d dev + %d test sentences" % (
            len(self.train.sentences),
            len(self.dev.sentences),
            len(self.test.sentences),
        )

    def get_all_sentences(self) -> Dataset:
        return ConcatDataset([self.train.sentences, self.dev.sentences, self.test.sentences])

    def get_all_tags(self) -> Dataset:
        return ConcatDataset([self.train.tags, self.dev.tags, self.test.tags])

    def make_tag_dictionary(self) -> Dict:

        # Make the tag dictionary
        tag_dictionary: Dict = dict()
        tag_dictionary['<START>'] = len(tag_dictionary)
        tag_dictionary['<END>'] = len(tag_dictionary)
        tag_dictionary['<PAD>'] = len(tag_dictionary)
        for sentence in self.get_all_tags():
            for token in sentence:
                if token not in tag_dictionary.keys():
                    tag_dictionary[token] = len(tag_dictionary)

        return tag_dictionary

    def calc_threshold_mean(self, features):
        """
        calculate the threshold for bucket by mean
        """
        lines_len = list(map(lambda t: len(t) + 1, features))
        average = int(sum(lines_len) / len(lines_len))
        lower_line = list(filter(lambda t: t < average, lines_len))
        upper_line = list(filter(lambda t: t >= average, lines_len))
        if len(lower_line) == 0:
            lower_average = 0
        else:
            lower_average = int(sum(lower_line) / len(lower_line))
        if len(upper_line) == 0:
            upper_average = 0
        else:
            upper_average = int(sum(upper_line) / len(upper_line))
        max_len = max(lines_len)
        return [lower_average, average, upper_average, max_len]#, 50, 65, 68, 70, 75, 80, 85, 90

    def construct_for_crf(self, dataset, tag_map):
        labels = dataset.tags
        labels = list(map(lambda t: ['<START>'] + list(t), labels))
        labels = list(map(lambda sent: [tag_map[t] for t in sent], labels))
        thresholds = self.calc_threshold_mean(dataset.sentences)
        label_size = len(tag_map)
        pad_feature = '<eof>'
        # pad_feature = f_map['<eof>']

        pad_label = tag_map['<PAD>']

        buckets = [[] for _ in range(len(thresholds))]
        for feature, label in zip(dataset.sentences, labels):
            # feature = list(map(lambda m: f_map.get(m.lower(), f_map['unk']), feature))
            cur_len = len(feature)
            idx = 0
            cur_len_1 = cur_len + 1  # label比feature多一个 <start>
            while thresholds[idx] < cur_len_1:
                idx += 1
            sent = feature + [pad_feature] * (thresholds[idx] - cur_len)
            tag = torch.LongTensor([label[ind] * label_size + label[ind + 1] for ind in range(0, cur_len)] + [
                label[cur_len] * label_size + pad_label] + [pad_label * label_size + pad_label] * (
                                           thresholds[idx] - cur_len_1))
            mask = torch.BoolTensor([1] * cur_len_1 + [0] * (thresholds[idx] - cur_len_1))
            buckets[idx].append(Sentence(sent, tag, mask, cur_len))
            # buckets[idx][0].append(feature + [pad_feature] * (thresholds[idx] - cur_len))
            # buckets[idx][1].append([label[ind] * label_size + label[ind + 1] for ind in range(0, cur_len)] + [
            #     label[cur_len] * label_size + pad_label] + [pad_label * label_size + pad_label] * (
            #                                thresholds[idx] - cur_len_1))
            # buckets[idx][2].append([1] * cur_len_1 + [0] * (thresholds[idx] - cur_len_1))

        # bucket_dataset = [DataFormat(bucket[0], torch.LongTensor(bucket[1]), torch.BoolTensor(bucket[2]))
        #                   for bucket in buckets if len(torch.LongTensor(bucket[1]).shape) > 0]
        # bucket_dataset = [CRFDataset(torch.LongTensor(bucket[0]), torch.LongTensor(bucket[1]), torch.BoolTensor(bucket[2]))
        #                   for bucket in buckets if len(torch.LongTensor(bucket[1]).shape) > 0]
        for buck in buckets:
            if len(buck) < 1:
                buckets.remove(buck)
        return buckets#bucket_dataset

    def construct_for_softmax(self, dataset, tag_map):
        """
        Construct bucket by mean for viterbi decode, word-level only
        """
        # encode and padding
        labels = list(map(lambda sent: [tag_map[t] for t in sent], dataset.tags))

        thresholds = self.calc_threshold_mean(dataset.sentences)
        pad_feature = '<eof>'
        pad_label = tag_map['<PAD>']

        buckets = [[] for _ in range(len(thresholds))]
        for feature, label in zip(dataset.sentences, labels):
            cur_len = len(feature)
            idx = 0

            while thresholds[idx] < cur_len:
                idx += 1

            sent = feature + [pad_feature] * (thresholds[idx] - cur_len)
            tag = torch.LongTensor(label + [pad_label] * (thresholds[idx] - cur_len))
            mask = torch.BoolTensor([1] * cur_len + [0] * (thresholds[idx] - cur_len))
            buckets[idx].append(Sentence(sent, tag, mask, cur_len))
            # buckets[idx][0].append(feature + [pad_feature] * (thresholds[idx] - cur_len))
            # buckets[idx][1].append(label + [pad_label] * (thresholds[idx] - cur_len))
            # buckets[idx][2].append([1] * cur_len + [0] * (thresholds[idx] - cur_len))

        # bucket_dataset = [DataFormat(bucket)
        #                   for bucket in buckets if len(torch.LongTensor(bucket[1].tag).shape) > 0]
        return buckets

    def construct_data(self, use_crf=True, tag_map=None, percentage=1):
        if percentage is not 1:
            length = len(self._train.sentences)
            import random
            idx = random.sample(range(length), int(percentage * length))
            self._train.sentences = [self._train.sentences[i] for i in idx]
            self._train.tags = [self._train.tags[i] for i in idx]
            # self._train.total_sentence_count = len(idx)

        # tag_map = self.make_tag_dictionary()
        if use_crf:
            train_dataset = self.construct_for_crf(self.train, tag_map)
            valid_dataset = self.construct_for_crf(self.dev, tag_map)
            test_dataset = self.construct_for_crf(self.test, tag_map)
        else:
            train_dataset = self.construct_for_softmax(self.train, tag_map)
            valid_dataset = self.construct_for_softmax(self.dev, tag_map)
            test_dataset = self.construct_for_softmax(self.test, tag_map)

        return train_dataset, valid_dataset, test_dataset


class DataFormat(Dataset):
    """Dataset Class for word-level model

    args:
        data_tensor (ins_num, seq_length): words
        label_tensor (ins_num, seq_length): labels
        mask_tensor (ins_num, seq_length): padding masks
    """
    def __init__(self, data_tensor):
        # assert len(data_tensor) == label_tensor.size(0)
        # assert len(data_tensor) == mask_tensor.size(0)
        # assert label_tensor.size(0) == mask_tensor.size(0)
        self.data_tensor = data_tensor
        # self.label_tensor = label_tensor
        # self.mask_tensor = mask_tensor

    def __getitem__(self, index):
        return self.data_tensor[index]#, self.label_tensor[index], self.mask_tensor[index]

    def __len__(self):
        return len(self.data_tensor[0].sentence)

class BertFormat(Dataset):
    """Dataset Class for word-level model

    args:
        data_tensor (ins_num, seq_length): words
        label_tensor (ins_num, seq_length): labels
        mask_tensor (ins_num, seq_length): padding masks
    """
    def __init__(self, data_tensor, mask_tensor):
        assert data_tensor.size(0) == mask_tensor.size(0)
        self.data_tensor = data_tensor
        self.mask_tensor = mask_tensor

    def __getitem__(self, index):
        return self.data_tensor[index], self.mask_tensor[index]

    def __len__(self):
        return len(self.data_tensor)


class Sentence:

    def __init__(self, sentence: List, tag: torch.LongTensor, mask: torch.BoolTensor, length: int):
        self.tokens = sentence
        self.tag = tag.unsqueeze(0)
        self.mask = mask.unsqueeze(0)
        self.length = length
        self.total_len = len(self.tokens)
        self.word_id: torch.LongTensor = None
        self.char_id: torch.LongTensor = None
        self.char_mask: torch.BoolTensor = None
        self.bertEmbedding: torch.FloatTensor = None
        self.tokenForBert = None

    def set_word(self, word_idx: torch.LongTensor):
        self.word_id = word_idx
        return self.word_id.cuda()

    def set_char(self, char_idx: torch.LongTensor, char_mask: torch.BoolTensor):
        self.char_id = char_idx
        self.char_mask = char_mask
        return self.get_char()

    def get_char(self):
        return self.char_id.cuda(), self.char_mask.cuda()

    def set_bert(self, bert_tmp):
        # pad_len = self.length - len(bert_tmp)
        # bert_tmp += [pad.unsqueeze(0)] * pad_len
        self.bertEmbedding = bert_tmp

        return self.bertEmbedding

    def tokenize(self):
        ''' For bert
        :return:
        '''
        if self.tokenForBert is None:
            self.tokenForBert = " ".join([t for t in self.tokens[:self.length]])

        return self.tokenForBert


