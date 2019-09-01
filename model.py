from typing import Union
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from dataloader import Encoder
from util import coalesce


class CharRNN(nn.Module):
    def __init__(self, vocab_size: int, embedding_size: int, hidden_size: int, padding_idx: int, n_layers: int=1):
        super(CharRNN, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.padding_idx = torch.tensor(padding_idx)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.padding_idx = self.padding_idx.to(self.device)

        self.embedder = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_size,
                                     padding_idx=self.padding_idx)
        self.rnn = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, num_layers=n_layers, batch_first=True)
        self.dense = nn.Linear(hidden_size, embedding_size)
        self.output_layer = nn.Linear(embedding_size, vocab_size)
        # Tie the weights of the output embeddings with the input embeddings
        self.output_layer.weight = self.embedder.weight
        self.loss_func = nn.CrossEntropyLoss()

        self.hidden = None
        self.init_rnn(self.rnn)

    def forward(self, x, lengths):
        batch_size, seq_length = x.size()
        out = self.embedder(x)
        # Initialize hidden states
        self.hidden = coalesce(self.hidden, self.init_hidden(batch_size))
        if self.hidden[0].size(1) > batch_size:
            warnings.warn(f'Expected batch_size {self.hidden[0].size(1)}, but received {batch_size} '
                          f'instead. Trimming the hidden state to match the batch size')
            self.hidden = (self.hidden[0][:, :batch_size, :].contiguous(),
                           self.hidden[1][:, :batch_size, :].contiguous())
        # RNNs expect a PackedSequence object unless you want to write a for loop yourself which is way slower
        out = nn.utils.rnn.pack_padded_sequence(out, lengths, batch_first=True)
        out, self.hidden = self.rnn(out, self.hidden)
        # Undo the packing operation
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=seq_length)
        out = out.contiguous()
        out = out.view(-1, out.size(2))  # Dimensions -> (Batch . Sequence) x Hidden
        out = self.dense(out)  # Dimensions -> (Batch . Sequence) x Embedding
        out = self.output_layer(out)  # Dimensions -> (Batch . Sequence) x Vocab
        return out.view(batch_size, seq_length, self.vocab_size)  # Dimensions -> Batch x Sequence x Vocab

    def loss(self, predictions, y, mask):
        predictions = predictions.view(-1, predictions.size(2))
        predictions *= torch.stack([mask] * predictions.size(1)).transpose(0, 1).float()
        return self.loss_func(predictions, y)

    @staticmethod
    def sample(probabilities):
        return int(np.random.choice(len(probabilities), size=1, p=probabilities.data.cpu().numpy())[0])

    def generate_text(self, encoder: Encoder, starting_seq: Union[list, str], sample_size: int):
        """
        Samples some text from the model
        :param encoder: The encoder to map tokens and ids into each other whenever required
        :param starting_seq: It should have at least one character. This str is run through the network first and the
        corresponding sentiments are recorded
        :param sample_size: The number to characters to generate besides the starting_seq characters
        :return: The generated ids (and not tokens) and their corresponding clusters
        """
        def _single_fwd_pass(char):
            o = self.forward(torch.tensor(char).view(1, 1).to(self.device), [1])
            o = o.view(self.vocab_size)
            o = F.softmax(o, dim=0)
            o = self.sample(o)
            return o

        starting_symbols = encoder.map_tokens_to_ids(starting_seq)
        starting_symbols_tensor = torch.tensor(starting_symbols).to(self.device)
        current_hidden_state = self.hidden
        self.reset_intermediate_vars()

        outputs = starting_symbols
        for x in starting_symbols_tensor:
            out = _single_fwd_pass(x)
        outputs.append(out)

        for i in range(sample_size):
            out = _single_fwd_pass(out)
            outputs.append(out)

        self.hidden = current_hidden_state
        return outputs

    def reset_intermediate_vars(self):
        self.hidden = None

    def detach_intermediate_vars(self):
        self.hidden = (self.hidden[0].detach(), self.hidden[1].detach())

    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)).to(self.device),
                Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)).to(self.device))

    @staticmethod
    def init_rnn(rnn):
        """
        Hidden to hidden weights -> Orthogonal initialization
        Input to hidden weights -> Xavier Initialization
        Bias -> Set to 1
        """
        for name, p in rnn.named_parameters():
            if name.startswith('weight_hh'):
                torch.nn.init.orthogonal_(p)
            elif name.startswith('weight_ih'):
                torch.nn.init.xavier_uniform_(p)
            elif 'bias' in name:
                torch.nn.init.constant_(p, 1)
