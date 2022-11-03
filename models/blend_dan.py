import torch
import torch.nn as nn
import torch.nn.functional as F


class RobertaDAN(nn.Module):

    def __init__(self,
                 vocab_size=10000,
                 embed_dim=300,
                 hidden_dim=256,
                 output_dim=3,
                 dropout=0.2,
                 embed_weight=None):
        super(RobertaDAN, self).__init__()

        self.embed = nn.Embedding(vocab_size, embed_dim)
        if embed_weight: self._load_embed(embed_weight)
        self.dropout1 = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(embed_dim)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        # the roberta output will be concat to this hidden input, thus we need to add 3 (number of output classes)
        # for computational efficiency, the forward result of roberta has beed precomputed
        # zipped along with the original input vector
        self.bn2 = nn.BatchNorm1d(hidden_dim + 3)
        self.fc2 = nn.Linear(hidden_dim + 3, output_dim)

    def _load_embed(self, embed_weight):
        self.embed.weight.data.copy_(embed_weight)
        self.embed.weight.requires_grad = False

    def forward(self, x):
        """
        roberta_label = x[:, -3:]
        x_copy = x[:, :-3]
        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        
        x, y_roberta = x[:, :-3], x[:, -3:]


        x = self.embed(x)
        x = x.mean(dim=1)
        x = self.dropout1(x)
        x = self.bn1(x)
        x = self.fc1(x)
        # here, we use the roberta model to provide generic sentiment feature
        x  = torch.cat((x, y_roberta), dim=1)
        x = self.dropout2(x)
        x = self.bn2(x)
        x = self.fc2(x)
        return x

