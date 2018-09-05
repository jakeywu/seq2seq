import os
import json
import random
import codecs
import numpy as np


class PrePareQaData(object):
    def __init__(self, conf, mode):
        self._mode = mode
        self._config = conf
        self._currPath = os.path.dirname(__file__)
        self._vocabDict = self.__load_chinese_vocab()
        self._sourceData = self.__read_dataset()
        self.PAD = 0
        self.SOS = 1
        self.EOS = 2

    def __load_chinese_vocab(self):
        cv = dict()
        with codecs.open(os.path.join(self._currPath, "data/chinese_vocab.txt"), "r", "utf8") as f:
            for i, line in enumerate(f.readlines()):
                cv[line.strip()] = i
        return cv

    def __read_dataset(self):
        if self._mode == "train":
            dataset_path = os.path.join(self._currPath, "data/trainset.txt")
        elif self._mode == "test":
            dataset_path = os.path.join(self._currPath, "data/testset.txt")
        else:
            raise Exception("mode must be in [train/test]")
        if not os.path.exists(dataset_path):
            raise Exception("path [{}] not exists".format(dataset_path))

        with codecs.open(dataset_path, "r", "utf8") as fp:
            dataset = fp.readlines()
            random.shuffle(dataset)
            return iter(dataset)

    def __word_to_id(self, dialogue):
        _id_lst = []
        for char in dialogue:
            _id = self._vocabDict.get(char, -1)
            if _id == -1:
                continue
            _id_lst.append(_id)
        return _id_lst

    def __parse_dialogue(self, dialogue_lst):
        encoder_id_lst = []
        decoder_id_lst = []
        for dialogue in dialogue_lst:
            dialogue = json.loads(dialogue)
            encoder_id_lst.append(self.__word_to_id(dialogue["Q"]))
            decoder_id_lst.append(self.__word_to_id(dialogue["A"]))
        return encoder_id_lst, decoder_id_lst

    def __padding_decoder_id(self, decoder_id_lst):
        max_len = max([len(item) for item in decoder_id_lst])
        for encoder_id in decoder_id_lst:
            encoder_id.extend((max_len-len(encoder_id)) * [self.PAD])

        lst = []
        for encoder_id in decoder_id_lst:
            lst.append([self.SOS] + encoder_id + [self.EOS])
        return lst

    def __padding_encoder_id(self, encoder_id_lst):
        max_len = max([len(item) for item in encoder_id_lst])
        for encoder_id in encoder_id_lst:
            encoder_id.extend((max_len - len(encoder_id)) * [self.PAD])

        return encoder_id_lst

    def __iter__(self):
        return self

    def __next__(self):
        dialogue_lst = []
        count = 0
        try:
            while count < self._config.batch_size:
                cur = next(self._sourceData)
                if not cur:
                    continue
                count += 1
                dialogue_lst.append(cur)
        except StopIteration as iter_exception:
            if count == 0:
                raise iter_exception

        encoder_input, decoder_target = self.__parse_dialogue(dialogue_lst)
        encoder_input = self.__padding_encoder_id(encoder_input)
        decoder_target = self.__padding_decoder_id(decoder_target)
        return np.array(encoder_input), np.array(decoder_target)
