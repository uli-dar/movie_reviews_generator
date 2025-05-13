import torch
import torch.nn as nn


class LSTM_LanguageModel(nn.Module):
    def __init__(self, hidden_dim, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.rnn = nn.LSTM(
            hidden_dim, hidden_dim, batch_first=True, num_layers=3, dropout=0.15
        )
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.projection = nn.Linear(hidden_dim, vocab_size)

        self.non_lin = nn.Tanh()
        self.dropout = nn.Dropout(p=0.15)

    def forward(self, input_batch: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(input_batch)  # [batch_size, seq_len, hidden_dim]
        output, _ = self.rnn(embeddings)  # [batch_size, seq_len, hidden_dim]
        output = self.dropout(
            self.linear(self.non_lin(output))
        )  # [batch_size, seq_len, hidden_dim]
        projection = self.projection(
            self.non_lin(output)
        )  # [batch_size, seq_len, vocab_size]
        return projection


class GRU_LanguageModel(nn.Module):
    def __init__(self, hidden_dim, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.rnn = nn.GRU(
            hidden_dim, hidden_dim, batch_first=True, num_layers=3, dropout=0.3
        )
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.projection = nn.Linear(hidden_dim, vocab_size)

        self.non_lin = nn.Tanh()
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, input_batch: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(input_batch)  # [batch_size, seq_len, hidden_dim]
        output, _ = self.rnn(embeddings)  # [batch_size, seq_len, hidden_dim]
        output = self.dropout(
            self.linear(self.non_lin(output))
        )  # [batch_size, seq_len, hidden_dim]
        projection = self.projection(
            self.non_lin(output)
        )  # [batch_size, seq_len, vocab_size]
        return projection
