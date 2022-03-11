import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=32, p_dropout=0.25):
        super(Embedding, self).__init__()
        self.relu = nn.LeakyReLU(inplace=True)
        self.dropout = nn.Dropout(p_dropout)

        self.w1 = nn.Linear(in_channels, mid_channels)
        self.b1 = nn.BatchNorm1d(mid_channels)
        self.w2 = nn.Linear(mid_channels, out_channels)
        self.b2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = self.dropout(self.relu(self.b1(self.w1(x))))
        x = self.dropout(self.relu(self.b2(self.w2(x))))

        return x