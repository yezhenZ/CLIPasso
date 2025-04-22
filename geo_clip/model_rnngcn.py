import torch
import torch.nn.functional as F
import torch.nn as nn
import math



class SimpleCNN(nn.Module):
    def __init__(self, input_channels=16, output_dim=128, shared_fc=None):
        super(SimpleCNN, self).__init__()

        # 定义卷积层部分
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=7, stride=1, padding=3),  # 输入：1通道，输出：16通道
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64x64 -> 32x32

            nn.Conv2d(16, 32, kernel_size=7, stride=1, padding=3),  # 输入：16通道，输出：32通道
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x32 -> 16x16

            nn.Conv2d(32, 64, kernel_size=7, stride=1, padding=3),  # 输入：32通道，输出：64通道
            nn.ReLU(),  # 添加ReLU激活函数
            nn.MaxPool2d(kernel_size=2, stride=2)  # 16x16 -> 8x8
        )

        # 展平操作，将卷积层输出展平成向量
        self.flatten = nn.Flatten()

        # 特征提取
        self.fc = nn.Sequential(
            nn.Linear(64 * 8 * 8, 16 * output_dim),
        )

        # 转化为坐标点
        self.fc2 = nn.Sequential(
            nn.ReLU(),  # 添加ReLU激活函数
            nn.Linear(16 * output_dim, 16 * 8),
            nn.Sigmoid()
        )

        # 使用共享全连接层
        self.shared_fc = shared_fc

    def forward(self, x):
        # 通过卷积层部分
        x = self.features(x)
        # 展平输出
        x = self.flatten(x)

        # 通过全连接层
        x1 = self.fc(x)
        # 使用共享全连接层（没有激活函数）
        x2 = self.shared_fc(x1)

        return x1, x2


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input_features, adj):
        # support = torch.mm(input_features, self.weight)
        output = torch.matmul(adj, input_features)
        # if self.bias is not None:
        #     return output + self.bias
        # else:
        #     return output
        return output

class GCN(nn.Module):
    def __init__(self, input_dim, output_dim, ctrlpt_head=None):
        super(GCN, self).__init__()
        self.gcn1 = GraphConvolution(input_dim, 64)
        self.gcn2 = GraphConvolution(64, 128)
        self.gcn3 = GraphConvolution(16, 8)
        self.linear1 = nn.Linear(input_dim, 64)
        self.linear2 = nn.Linear(64, 8)

        # Latent to 4 control points (4×2=8)
        self.ctrlpt_head = ctrlpt_head    # 使用共享全连接层

    def forward(self, X, adj):
        X=self.gcn1(X,adj)
        # X = self.linear1(X)
        # X = F.relu(X)  # Graph convolution + ReLU
        # X = self.linear2(X)  # Output layer
        X = F.relu(X)
        X=self.ctrlpt_head(X)
        # 展平输出
        X = torch.sigmoid(X)  # Graph convolution + ReLU
        return X

