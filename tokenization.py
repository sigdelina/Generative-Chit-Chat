import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import youtokentome as yttm
import torch.nn.functional as F


class TrainDataset(Dataset):

    def __init__(self, path_to_file, sep='\t'):
        super().__init__()

        self.path_to_file = path_to_file
        self.sep = sep

        self.data = self.load_data()
        self.train, self.val, self.test = self.split_dataset()
        self.dataframe = pd.DataFrame({'question': self.data[0], 'answer': self.data[1]})

    def load_data(self):
        data_seq = []
        question, answer = [], []
        counter = 0
        with open(self.path_to_file, ) as f:
            for line in f:
                if counter == 400000:
                    break
                data_seq.append(line.strip().split(self.sep))
                counter += 1

        return data_seq

    def prepare_tokenization(self):

        with open('for_bpe.txt', 'w', encoding='utf-8') as f:
            for que, answ in self.train:
                f.write(que + '\t' + answ + '\n')

    def split_dataset(self):

        train, val, test = np.split(np.array(self.data), [int(.8 * len(self.data)), int(.9 * len(self.data))])

        return train, val, test


class Tokenizer:

    def __init__(self, model_path, max_length):
        self.tokenizer = yttm.BPE(model=model_path)

        self.tokenizer.pad_token = '<pad>'
        self.pad_index = 0
        self.max_length = max_length

    def tokenize(self, text):
        """
        В этом методе нужно разделить строку текста на токены
        """
        ...

        return self.tokenizer.encode(text, bos=True, eos=True)

    def padding(self, tokens_indices):
        """
        В этом методе нужно сделать длину tokens_indices равной self.max_length
        """
        ...
        padded_seq = F.pad(torch.tensor(tokens_indices), (0, self.max_length - len(tokens_indices)),
                           value=self.pad_index)

        return padded_seq

    def __call__(self, text):
        """
        В этом методе нужно перевести строку с текстом в вектор с индексами слов нужно размера (self.max_length)
        """
        ...

        return torch.LongTensor(self.padding(self.tokenize(text)))

    def no_eos(self, text):
        return torch.LongTensor(self.padding(self.tokenizer.encode(text, bos=True, eos=False)))

    def collate(self, batch):
        question = torch.stack([self.__call__(el[0]) for el in batch])

        answer = torch.stack([self.__call__(el[1]) for el in batch])

        answer_no_eos = torch.stack([self.no_eos(el[1]) for el in batch])

        return question, answer, answer_no_eos