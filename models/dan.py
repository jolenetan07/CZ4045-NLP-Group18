import torch
import torch.nn as nn
import torch.nn.functional as F


class DAN(nn.Module):

    def __init__(self,
                 vocab_size=10000,
                 embed_dim=300,
                 hidden_dim=256,
                 output_dim=2,
                 dropout=0.2,
                 embed_weight=None):
        super(DAN, self).__init__()

        self.embed = nn.Embedding(vocab_size, embed_dim)
        if embed_weight: self._load_embed(embed_weight)
        self.dropout1 = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(embed_dim)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def _load_embed(self, embed_weight):
        self.embed.weight.data.copy_(embed_weight)
        self.embed.weight.requires_grad = False

    def forward(self, batch):
        text = batch.text
        label = batch.label
        x = self.embed(text)
        x = x.mean(dim=0)
        x = self.dropout1(x)
        x = self.bn1(x)
        x = self.fc1(x)
        x = self.dropout2(x)
        x = self.bn2(x)
        x = self.fc2(x)
        return x
