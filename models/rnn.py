# -*- coding:utf-8 -*-
"""
@file name  : rnn.py
@author     : XiaoHeng
@date       : 2025-2-10
@brief      : RNN modeling
"""
import os
import sys
import time
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import platform
from torch.utils.data import DataLoader
import torch.nn.functional as F


BASE_DIR = os.path.dirname(__file__)
PRJ_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.append(PRJ_DIR)


class LSTMTextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, **kwargs):
        super(LSTMTextClassifier, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # bidirectional=True: for get the Bidirectional Recurrent Neural Network
        self.encoder = nn.LSTM(embed_size, num_hiddens, num_layers=num_layers, bidirectional=True)
        self.decoder = nn.Linear(4 * num_hiddens, 2)  # 一个output向量长度是2倍的hidden size，有两个output拼接，所以是4倍
        # https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html#torch.nn.LSTM

    def forward(self, inputs):
        # inputs（batch_size，time_steps）
        # Because LSTM networks require that the first dimension of their input is the time dimension, the input is transposed before obtaining the token representation.
        # output（time_steps，batch_size，vocabulary_vector_dimension）
        embeddings = self.embedding(inputs.T)
        self.encoder.flatten_parameters()
        # Returns the hidden state of the previous hidden layer at different time steps,
        # outputs（time_steps，batch_size，2*hidden_unit_numbers）
        outputs, _ = self.encoder(embeddings)
        # Concatenate the hidden states of the initial and final time steps as the input to the fully connected layer,
        # shape（batch_size，4*hidden_unit_numbers）
        encoding = torch.cat((outputs[0], outputs[-1]), dim=1)
        outs = self.decoder(encoding)
        return outs


class RNNTextClassifier(nn.Module):
    def __init__(self, inp_size, hidden_size, n_class, layer_num, vocab_len, device):
        super(RNNTextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_len, inp_size)

        self.device = device
        self.hidden_size = hidden_size
        self.layer_num = layer_num
        self.rnn = nn.RNN(input_size=inp_size, hidden_size=hidden_size, num_layers=layer_num, batch_first=True)
        self.fc = nn.Linear(hidden_size, n_class, bias=True)

    def forward(self, x):

        x_embed = self.embedding(x)  # [batch_size, max_len] -> [batch_size, text_len, embed_len]
        outputs, hidden = self.rnn(x_embed)
        last_hidden = hidden[-1].squeeze(0)  # [num_layers, bs, hidden_size]  ->  [bs, hidden_size]
        fc_output = self.fc(last_hidden)

        return fc_output

    def forward_bak(self, x):

        x_embed = self.embedding(x)  # [batch_size, max_len] -> [batch_size, text_len, embed_len]

        bs_, text_len, embed_len = x_embed.shape

        hidden_init = self.init_hidden(bs_)
        outputs, hidden = self.rnn(x_embed, hidden_init)
        # Extract the last hidden state
        last_hidden = hidden[-1].squeeze(0)  # [num_layers, bs, hidden_size]  ->  [bs, hidden_size]
        last_output = outputs[:, -1, :].squeeze(0)  # bs, sequence len, hidden_size]  ->  [bs, hidden_size]
        fc_output = self.fc(last_hidden)

        return fc_output

    def init_hidden(self, batch_size):
        # https://pytorch.org/docs/stable/generated/torch.nn.RNN.html#torch.nn.RNN
        hidden = torch.zeros(self.layer_num, batch_size, self.hidden_size)  # (D∗num_layers, N, H_out)
        hidden = hidden.to(self.device)
        return hidden


if __name__ == "__main__":
    # Example usage
    input_size = 768  # Size of input features (e.g., word embeddings)
    hidden_size = 128  # Number of hidden units in the RNN
    layer_num = 3  # Number of hidden layer in the RNN
    output_size = 4  # Number of output classes (binary classification)
    vocab_len = 20000

    # Create an instance of the RNNTextClassifier
    model = RNNTextClassifier(input_size, hidden_size, output_size, layer_num, vocab_len, 'cpu')

    batch_size = 16
    sequence_length = 200
    input_token_list = torch.randint(vocab_len, (batch_size, sequence_length))

    output = model(input_token_list)
    print(output.shape)