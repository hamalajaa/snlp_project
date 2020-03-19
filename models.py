import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(self, hps, vocab_size, dimensions):
        super(LSTM, self).__init__()
        self.N = dimensions
        self.embeds = nn.Embedding(vocab_size, 10)
        # Recurrent layer
        self.lstm = nn.LSTM(10, hps.lstm_h_dim, 2)

        # Output layer
        self.l_out = nn.Linear(in_features=hps.lstm_h_dim,
                               out_features=1)

    def forward(self, x):

        # Embedding
        #print("x oli", x.shape)
        x = self.embeds(x)
        #print("Before view", x.shape)
        x = x.view((x.shape[0],self.N - 1, -1))
        #print('After embed', x.shape)
        # RNN returns output and last hidden state
        x, (h, c) = self.lstm(x)

        # Output layer
        x = self.l_out(x)

        return x
