import torch
import torch.nn as nn


class MusicRNN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, unique_notes, seq_len, rnn='LSTM'):
        assert rnn in ['LSTM', 'GRU']
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(unique_notes+1, embedding_dim)
        if rnn == 'LSTM':
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=3, dropout=0.2)
        else:
            self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers=3, dropout=0.2)
        self.linear = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * seq_len, 512),
            nn.LeakyReLU(),
            nn.Linear(512, unique_notes+1)
        )

    def forward(self, x):
        batch_size, _ = x.shape
        embeds = self.embeddings(x)
        rnn_out, _ = self.rnn(embeds.view(self.seq_len, batch_size, -1))
        outputs = self.linear(rnn_out.view(batch_size, -1))
        return outputs
