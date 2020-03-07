
import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(self, hps, vocab_size):
        super(LSTM, self).__init__()

        # Recurrent layer
        self.lstm = nn.LSTM(vocab_size, hps.lstm_h_dim, 2)
        
        # Output layer
        self.l_out = nn.Linear(in_features=hps.lstm_h_dim,
                            out_features=vocab_size)
        
    def forward(self, x):

        d, n, v = x.shape

        # RNN returns output and last hidden state
        x, (h, c) = self.lstm(x)
        
        # Flatten output for feed-forward layer
        #x = x.view(-1, self.lstm.hidden_size)
        
        # Output layer
        x = self.l_out(x)
        
        return x




