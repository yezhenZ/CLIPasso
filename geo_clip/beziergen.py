import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.distributions as dist
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pad_sequence

# from bezierloss import BezierLoss


class SimpleBezierAE(nn.Module):
    def __init__(self, n_input=2, n_hidden=128, n_layer=2, n_latent=128, dropout=0.1, bidirectional=True):
        super().__init__()

        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_layer = n_layer
        self.n_latent = n_latent
        self.bidirectional = 2 if bidirectional else 1

        # LSTM Encoder
        self.encoder = nn.LSTM(
            input_size=n_input,
            hidden_size=n_hidden,
            batch_first=True,
            dropout=dropout,
            num_layers=n_layer,
            bidirectional=bidirectional
        )

        # Hidden to latent 
        self.hc_project = nn.Linear(self.bidirectional  * n_hidden, n_latent)

        # Latent to  control points (4×2=8)
        self.ctrlpt_head = nn.Linear(n_latent, 4 * 2)

    def forward(self, x):
        # x: [batch, len_seq, 2]
        out, (h_n, c_n) = self.encoder(x)  # h_n: [n_layer * num_directions, batch, hidden]
        # 获取最后一层的 forward & backward 的 hidden
        h_n = h_n.view(self.n_layer, self.bidirectional, -1, self.n_hidden)
        c_n = c_n.view(self.n_layer, self.bidirectional, -1, self.n_hidden)

        h_last = h_n[-1]  # [2, batch, hidden]
        c_last = c_n[-1]  # [2, batch, hidden]

        H = torch.cat([h_last[0], h_last[1]], dim=1)  # [batch, 2*hidden]

        # C = torch.cat([c_last[0], c_last[1]], dim=1)  # [batch, 2*hidden]
        feature=self.hc_project(H)
        latent = F.relu(self.hc_project(H))  # [batch, latent]

        ctrl_pts = self.ctrlpt_head(latent).view(-1, 4, 2)  # [batch, 4, 2]
        ctrl_pts=torch.sigmoid(ctrl_pts)
        return ctrl_pts,feature
    def save(self, path):
        """保存模型权重"""
        torch.save({
            'state_dict': self.state_dict()
        }, path)
