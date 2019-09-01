import os
import shutil
import string
import tarfile
import tempfile
import urllib.request
from abc import abstractmethod
from math import ceil
from typing import Union, List
from zipfile import ZipFile

import torch
from torch.utils.data import Dataset

from config import text8config, ptbconfig, hutter_prize_config
from util import print_tokens

END_OF_SENTENCE_TOKEN = '<EOS>'
OUT_OF_VOCAB_TOKEN = '<OOV>'
PADDING_TOKEN = '<PAD>'


class Encoder:
    def __init__(self):
        self._vocab = {}
        self._inverse_vocab = {}
        self._current_id = 0
        self._init_vocab()

    def _init_vocab(self) -> None:
        self.add_to_vocab(PADDING_TOKEN)
        self.add_to_vocab(END_OF_SENTENCE_TOKEN)
        self.add_to_vocab(OUT_OF_VOCAB_TOKEN)

    def add_to_vocab(self, token: str) -> None:
        if token not in self._vocab:
            self._vocab[token] = self._current_id
            self._inverse_vocab[self._current_id] = token
            self._current_id += 1

    @property
    def vocabulary_size(self) -> int:
        return len(self._vocab)

    def token_to_id(self, token: str) -> int:
        return self._vocab.get(token, self._vocab[OUT_OF_VOCAB_TOKEN])

    def id_to_token(self, id_: int) -> str:
        return self._inverse_vocab.get(id_, OUT_OF_VOCAB_TOKEN)

    def map_tokens_to_ids(self, tokens: Union[list, str]) -> list:
        return [self.token_to_id(token) for token in tokens]

    def map_ids_to_tokens(self, ids: List[int]) -> list:
        return [self.id_to_token(i) for i in ids]


class LanguageModelingDataset(Dataset):
    def __init__(self, batch_size, sequence_length):
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.X = []
        self.y = []
        self.lengths = []
        self.encoder = Encoder()
        self.init_vocab()
        self.load_data()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index: int):
        return self.X[index], self.y[index], self.lengths[index]

    @abstractmethod
    def init_vocab(self):
        pass

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def download_dataset(self):
        pass

    @staticmethod
    def download_file(url: str, file_path, flags: str) -> None:
        data = urllib.request.urlopen(url).read()
        with open(file_path, flags) as f:
            f.write(data)

    def _prepare_data(self, data):
        """
        Assumes that the data is one long string which has to be divided into batches containing examples of
        length = self.sequence_length such that ith example in a batch is the continuation of the ith example
        from the previous batch
        """
        # Divide data into self.batch_size number of examples
        x = self._split_example(data, number_of_splits=self.batch_size)
        x = self._tokenize(x)
        y = self._create_target_sequences(x)
        # Now x has a single batch and each example in that batch is VERY long
        # We'll now split each example into n parts where each part is of length self.sequence_length
        # As a result, each example will be replaced by a list of these parts
        x = [self._split_example(example, length_of_each_split=self.sequence_length) for example in x]
        y = [self._split_example(example, length_of_each_split=self.sequence_length) for example in y]
        # Now we can finally construct batches by rearranging data
        # We rearrange data such that ith seq. in each batch is the continuation of ith seq. form the previous batch
        max_number_of_splits = max((len(i) for i in x))
        for i in range(max_number_of_splits):
            for j in range(self.batch_size):
                if i < len(x[j]):
                    self.X.append(x[j][i])
                    self.y.append(y[j][i])
                    self.lengths.append(len(x[j][i]))
        self.X = self._pad_data(self.X)
        self.y = self._pad_data(self.y)
        self.X = self._convert_to_tensors(self.X)
        self.y = self._convert_to_tensors(self.y)

    def _split_example(self, data, *, number_of_splits=None, length_of_each_split=None):
        """
        Divides the string into n parts. If it's not possible to divide the string into n EQUAL parts,
        then nth part will be smaller than the first n-1 parts(which will all be of equal lengths)
        :param data: The string to be divided into n parts
        :param number_of_splits: The number of parts to divide the string into
        """
        if not number_of_splits and not length_of_each_split:
            raise ValueError('At least one of the two keyword arguments must be provided')
        split_length = length_of_each_split or ceil(len(data) / number_of_splits)
        num_splits = number_of_splits or ceil(len(data) / split_length)
        return [data[i * split_length: min(len(data), (i + 1) * split_length)] for i in range(num_splits)]

    def _create_target_sequences(self, data: list):
        """Creates target values for language modeling by shifting sequences to the right by one"""
        return [x[1:] + [self.encoder.token_to_id(END_OF_SENTENCE_TOKEN)] for x in data]

    def _pad_data(self, data: list):
        """Pads each sequence in the list to make its length equal to self.sequence_length"""
        return [x + [self.encoder.token_to_id(PADDING_TOKEN)] * (self.sequence_length - len(x)) for x in data]

    def _tokenize(self, data: list):
        """Maps characters to integers"""
        return [self.encoder.map_tokens_to_ids(x) for x in data]

    def _convert_to_tensors(self, data: list):
        return [torch.tensor(x) for x in data]


class Text8Dataset(LanguageModelingDataset):
    def __init__(self, mode='train', batch_size=128, data_path=None, sequence_length=100, num_test_chars=5000000):
        self.mode = mode.lower()
        self.data_path = data_path or os.path.join('data', 'text8')
        self.num_test_chars = num_test_chars
        super(Text8Dataset, self).__init__(batch_size=batch_size, sequence_length=sequence_length)

    def init_vocab(self):
        for i in string.ascii_lowercase + ' ':
            self.encoder.add_to_vocab(i)

    def load_data(self):
        if not os.path.exists(self.data_path):
            self.download_dataset()
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = f.read()
        data = self._get_the_dataset_split(data)
        self._prepare_data(data)

    def download_dataset(self):
        url = 'http://mattmahoney.net/dc/text8.zip'
        _, tmp_file_path = tempfile.mkstemp(dir='.')
        print('Downloading text8 dataset')
        self.download_file(url, tmp_file_path, 'wb')
        print('Downloaded. Extracting text8 form zipfile')
        ZipFile(tmp_file_path).extractall()
        print('Data extracted. Cleaning up and moving the data files to data directory')
        if not os.path.exists('data') or not os.path.isdir('data'):
            os.mkdir('data')
        shutil.move('text8', self.data_path)
        os.remove(tmp_file_path)

    def _get_the_dataset_split(self, data: str):
        train_data = data[: -2 * self.num_test_chars]
        valid_data = data[-2 * self.num_test_chars: -self.num_test_chars]
        test_data = data[-self.num_test_chars:]
        if self.mode == 'train':
            return train_data
        elif self.mode == 'valid':
            return valid_data
        elif self.mode == 'test':
            return test_data


class PTBCharDataset(LanguageModelingDataset):
    def __init__(self, mode='train', batch_size=128, data_path=None, sequence_length=100):
        self.mode = mode
        self.data_path = data_path or os.path.join('data', 'ptb.char.' + self.mode + '.txt')
        super(PTBCharDataset, self).__init__(batch_size=batch_size, sequence_length=sequence_length)

    def init_vocab(self):
        for i in string.ascii_lowercase:
            self.encoder.add_to_vocab(i)
        for i in range(10):
            self.encoder.add_to_vocab(str(i))
        for i in ['N', '#', '\\', '&', '-', "'", ' ', '.', '/', '$', '_', '<', '*', '>']:
            self.encoder.add_to_vocab(i)

    def load_data(self):
        if not os.path.exists(self.data_path):
            self.download_dataset()
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = f.readlines()
        data = self._transform_data(data)
        self._prepare_data(data)

    def download_dataset(self):
        url = 'http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz'
        _, tmp_file_path = tempfile.mkstemp(dir='.')
        print('Downloading PTB dataset')
        self.download_file(url, tmp_file_path, 'wb')
        print('Downloaded. Extracting PTB form tgz file')
        tar = tarfile.open(tmp_file_path)
        tar.extractall()
        tar.close()
        print('Data extracted. Cleaning up and moving the data files to data directory')
        os.remove(tmp_file_path)
        if not os.path.exists('data') or not os.path.isdir('data'):
            os.mkdir('data')
        for ptb_data_split in ['train', 'valid', 'test']:
            shutil.move(os.path.join('simple-examples', 'data', 'ptb.char.' + ptb_data_split + '.txt'),
                        os.path.join('data', 'ptb.char.' + ptb_data_split + '.txt'))
        shutil.rmtree('simple-examples')

    def _transform_data(self, data: list):
        """
        First:  't h i s _ i s _ w i e r d\n' => list('this is weird')
        Then flatten out the list
        """
        data = [list(''.join(x[:-1].split(' ')).replace('_', ' ')) for x in data[:-1]]
        return [j for i in data for j in i]


class HutterPrizeDataset(LanguageModelingDataset):
    def __init__(self, mode='train', batch_size=128, data_path=None, sequence_length=100, num_test_chars=5000000):
        self.mode = mode.lower()
        self.data_path = data_path or os.path.join('data', 'enwik8')
        self.num_test_chars = num_test_chars
        super(HutterPrizeDataset, self).__init__(batch_size=batch_size, sequence_length=sequence_length)

    def init_vocab(self):
        if not os.path.exists(self.data_path):
            self.download_dataset()
        with open(self.data_path, 'rb') as f:
            data = f.read()
        for i in set(data):
            self.encoder.add_to_vocab(i)

    def load_data(self):
        if not os.path.exists(self.data_path):
            self.download_dataset()
        with open(self.data_path, 'rb') as f:
            data = f.read()
        data = self._get_the_dataset_split(data)
        self._prepare_data(data)

    def download_dataset(self):
        url = 'http://mattmahoney.net/dc/enwik8.zip'
        _, tmp_file_path = tempfile.mkstemp(dir='.')
        print('Downloading Hutter Prize (enwik8) dataset')
        self.download_file(url, tmp_file_path, 'wb')
        print('Downloaded. Extracting enwik8 form zipfile')
        ZipFile(tmp_file_path).extractall()
        print('Data extracted. Cleaning up and moving the data files to data directory')
        if not os.path.exists('data') or not os.path.isdir('data'):
            os.mkdir('data')
        shutil.move('enwik8', self.data_path)
        os.remove(tmp_file_path)

    def _get_the_dataset_split(self, data: bytes):
        train_data = data[: -2 * self.num_test_chars]
        valid_data = data[-2 * self.num_test_chars: -self.num_test_chars]
        test_data = data[-self.num_test_chars:]
        if self.mode == 'train':
            return train_data
        elif self.mode == 'valid':
            return valid_data
        elif self.mode == 'test':
            return test_data


class DatasetFactory:
    @staticmethod
    def get_batch_size(config, mode):
        if mode.lower() == 'train':
            return config['batch_size']
        return 1

    @staticmethod
    def get_dataset(dataset: str, mode: str) -> LanguageModelingDataset:
        if dataset.lower() == 'text8':
            return Text8Dataset(mode=mode.lower(),
                                batch_size=DatasetFactory.get_batch_size(text8config, mode),
                                sequence_length=text8config['sequence_length'])
        elif dataset.lower() == 'ptb':
            return PTBCharDataset(mode=mode.lower(),
                                  batch_size=DatasetFactory.get_batch_size(text8config, mode),
                                  sequence_length=ptbconfig['sequence_length'])
        elif dataset.lower() in ['hutter', 'hutter_prize', 'enwik8']:
            return HutterPrizeDataset(mode=mode.lower(),
                                      batch_size=DatasetFactory.get_batch_size(text8config, mode),
                                      sequence_length=hutter_prize_config['sequence_length'])
        else:
            raise ValueError('Invalid dataset name provided for getting dataset')

    @staticmethod
    def get_config(dataset: str):
        if dataset.lower() == 'text8':
            return text8config
        elif dataset.lower() == 'ptb':
            return ptbconfig
        elif dataset.lower() in ['hutter', 'hutter_prize', 'enwik8']:
            return hutter_prize_config
        else:
            raise ValueError('Invalid dataset name provided for getting configuration')


if __name__ == '__main__':
    ds = Text8Dataset('train', 128, sequence_length=50, num_test_chars=49997000)
    # ds = Text8Dataset('valid', 5000000, 1, 5000000)
    # ds = Text8Dataset('train')
    print(ds.X[0])
    print(ds.X[-1])
    print(ds.X[-2])
    print_tokens(ds.encoder.map_ids_to_tokens(ds.X[0].data.numpy()))
    print_tokens(ds.encoder.map_ids_to_tokens(ds.y[0].data.numpy()))
    print_tokens(ds.encoder.map_ids_to_tokens(ds.X[-1].data.numpy()))
    print_tokens(ds.encoder.map_ids_to_tokens(ds.y[-1].data.numpy()))
    print_tokens(ds.encoder.map_ids_to_tokens(ds.X[-2].data.numpy()))
    print_tokens(ds.encoder.map_ids_to_tokens(ds.y[-2].data.numpy()))
