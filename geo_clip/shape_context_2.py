import numpy as np
import torch
from math import comb as n_over_k  # Comb(n, k) 计算组合数
from matplotlib import pyplot as plt
import math
import torch.nn.functional as F
import time
from scipy.optimize import linear_sum_assignment
import os



class ShapeContext_2(object):
    """
    给定图像中的一个点，计算所有其他点相对于该点的角度。
    半径和角度存储在一个 "形状矩阵" 中，矩阵的维度为: radial_bins x angle_bins。
    每个元素 (i, j) 存储了对于一个给定的点，落在该半径bin和角度bin的点的数量。
    """

    def __init__(self, nbins_r=6, nbins_theta=12, r_inner=0.001, r_outer=2.5, wlog=False,device=torch.device('cuda')):
        self.nbins_r = nbins_r  # 径向方向的bin数
        self.nbins_theta = nbins_theta  # 角度方向的bin数
        self.r_inner = r_inner  # 内半径
        self.r_outer = r_outer  # 外半径
        self.nbins = nbins_theta * nbins_r  # 总的bin数
        self.wlog = wlog  # 是否使用log10(r) 或 归一化处理
        self.device = device
    def distM(self, x):
        """
        计算距离矩阵

        参数:
    -------
    x: 形状为 (batch_size, num_points, 2) 的张量，表示批量点集

    返回:
    --------
    result: 形状为 (batch_size, num_points, num_points) 的张量，表示每个批次的欧几里得距离矩阵
        """
        # 获取输入的形状
        batch_size, num_points, _ = x.shape
        x1 = x.unsqueeze(2)  # (batch_size, num_points, 1, 2)
        x2 = x.unsqueeze(1)  # (batch_size, 1, num_points, 2)

        dist_matrix = torch.norm(x1 - x2, dim=-1)  # 计算 L2 范数 (欧几里得距离)

        return dist_matrix  # (batch_size, num_points, num_points)

    def angleM(self, x):
        """
           计算角度矩阵
           参数:
           -------
           x: 形状为 (batch_size, num_points, 2) 的张量，每个点 (x, y)
           返回:
           --------
           result: 形状为 (batch_size, num_points, num_points) 的角度矩阵
           """
        batch_size, num_points, _ = x.shape  # (batch_size, num_points, 2)

        # 扩展维度，构造所有点的两两组合
        x1 = x.unsqueeze(2)  # (batch_size, num_points, 1, 2)
        x2 = x.unsqueeze(1)  # (batch_size, 1, num_points, 2)

        # 计算坐标差
        delta_x = x2[..., 0] - x1[..., 0]  # (batch_size, num_points, num_points)
        delta_y = x2[..., 1] - x1[..., 1]  # (batch_size, num_points, num_points)
        eps = 1e-6
        # 计算角度 (arctan2 计算 y/x 的角度，自动处理正负象限)
        angles = torch.atan2(delta_y+eps, delta_x+eps)  # 角度矩阵 (batch_size, num_points, num_points)

        return angles

    def compute(self, points):
        # points的形状为 (batch_size, num_points, 2)
        batch_size, num_points, _ = points.shape
        # 计算距离矩阵(batch_size, num_points, num_points)
        r_array = self.distM(points).to(self.device)

        # 使用对数尺度或归一化
        if self.wlog:
            r_array_n = torch.log10(r_array + 1)
        else:
            #这里要排除自距离？
            #除以均值，减少缩放敏感性
            # print(r_array)
            mean_dist = r_array.mean(dim=(1, 2), keepdim=True)  # 形状为 (batch_size, 1, 1)
            r_array_n = r_array / mean_dist
            # print(mean_dist.shape)
            # print(r_array_n)
        log_r_array_n = torch.log(r_array_n+1e-10).unsqueeze(-1)  # [batch_size,n,n, 1]

        # 计算径向分箱边界
        # r_bin_edges = torch.logspace(torch.log(torch.tensor(self.r_inner)),torch.log(torch.tensor(self.r_outer)), self.nbins_r + 1).to(self.device)
        r_bin_edges= torch.linspace(torch.tensor(self.r_inner), torch.tensor(self.r_outer), self.nbins_r + 1).to(self.device)

        #计算分箱中心
        centers = torch.sqrt(r_bin_edges[:-1] * r_bin_edges[1:])

        log_centers = torch.log(centers).unsqueeze(0)  # [1, nbins]
        logits = -5.*torch.abs(log_r_array_n - log_centers)  # [b,n,N, nbins]
        r_soft_bins = torch.softmax(logits, dim=-1)  # [b,n,N, nbins]

        # print(r_bin_edges)
        # print(f"the 06  is {r_array_n[0][1]}")
        # # print(f"the logits are {logits[0][2]}")
        # print(f"the new bins 06 are {r_soft_bins[0][1]}")
        #
        # print(f"the 08  is {r_array_n[0][0]}")
        # print(f"the new bins 08 are {r_soft_bins[0][0]}")

        #角度
        # 计算角度矩阵
        theta_array = self.angleM(points).to(self.device)
        # 将角度归一化到[0, 2*pi]区间
        # theta_array_2pi = theta_array + 2 * torch.pi * (theta_array < 0)
        theta_array_2pi = theta_array % (2 * torch.pi)  # [0, 2π]
        # 计算角度分箱边界
        theta_bin_edges = torch.linspace(0, 2 * torch.pi, self.nbins_theta + 1).to(self.device)

        # 计算角度和径向的分箱概率并进行采样
        theta_samples = self.compute_bin_probs(theta_array_2pi, theta_bin_edges)
        r_samples = self.compute_bin_probs(r_array_n, r_bin_edges)
        # print(f"the old bins 00  are {theta_samples[0][0]}")
        # print(f"the old bins 10  are {theta_samples[1][0]}")

        histogram = torch.einsum('acbi,acbj->acij', r_samples, theta_samples)  # (batchsize,num_points,num_r_bins, num_theta_bins)
        histogram_rtheta=histogram.view(batch_size, num_points, -1)
        # print(f"the histogram bins 00   are {histogram[0][0]}")
        # print(f"the histogram bins 10   are {histogram[1][0]}")

        return histogram_rtheta,histogram
    def compute_bin_probs(self,data, bin_edges, temperature=0.001):
        """
        计算数据点属于每个分箱的概率。

        参数:
            data: 输入数据，角度矩阵或径向矩阵形状为 (num_curves, num_points, num_points)
            bin_edges: 分箱边界，形状为 (num_bins + 1,)
            temperature: Gumbel-Softmax 的温度参数

        返回:
            samples: 采样结果，形状与 probs 相同
        """
        num_bins = bin_edges.shape[0] - 1  # 分箱数量
        data = data.unsqueeze(-1)  # (num_curves, num_points, num_points, 1)

        # 计算数据点到每个分箱边界的距离
        lower_edges = bin_edges[:-1].unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, num_bins)
        upper_edges = bin_edges[1:].unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, num_bins)

        # 使用 softmax 平滑分箱概率
        logits = -torch.abs(data - (lower_edges + upper_edges) / 2.0 + 1e-10)

        samples = F.softmax(logits / temperature, dim=-1)
        # print(f"the 02 samples result  is {samples[0][2]}")
        # print(f"the 16  samples is {samples[1][6]}")

        return samples

    def compute_cost_matrix(self,histograms1, histograms2):
        """
        计算代价矩阵,计算两个直方图之间的卡方距离
        :param histograms1: 轮廓1的形状上下文直方图矩阵 (N, K):N采样点个数
        :param histograms2: 轮廓2的形状上下文直方图矩阵 (M, K)：M采样点个数
        :return: 代价矩阵 (N, M)
        """
        eps = 1e-10  # 避免除以零

        # (N, 1, K) - (1, M, K) -> (N, M, K)
        numerator = (histograms1[:, None, :] - histograms2[None, :, :]) ** 2
        denominator = histograms1[:, None, :] + histograms2[None, :, :] + eps

        # 按照 K 维度求和 (N, M, K) -> (N, M)
        cost_matrix = 0.5*torch.sum(numerator / denominator,dim=2)
        # print(cost_matrix.shape)

        return cost_matrix

    def hungarian_matching(self,cost_matrix):
        """
        使用匈牙利算法找到最优匹配
        :param cost_matrix: 代价矩阵 (N, M)
        :return: 匹配索引和总代价
        """
        cost_matrix_normalized = cost_matrix / cost_matrix.max()
        row_ind, col_ind = linear_sum_assignment(cost_matrix_normalized.detach().cpu().numpy() )
        total_cost = cost_matrix_normalized[row_ind, col_ind].sum()
        return (row_ind, col_ind), total_cost

    # 计算两条曲线之间的相似度
    def compute_similarity(self,shape_context_histograms):
        """
        计算曲线之间的相似度
        :shape_context_histograms: 曲线矩阵的形状上下文直方图矩阵 (k, K)
        :return: 相似度距离
        """
        # 计算所有曲线对之间的相似度矩阵
        n_curves = len(shape_context_histograms)  # 曲线的数量
        similarity_matrix = torch.zeros((n_curves, n_curves))  # 相似度矩阵
        for i in range(n_curves):
            for j in range(n_curves):
                # 获取两条曲线的形状上下文直方图
                histograms1 = shape_context_histograms[i].clone()
                histograms2 = shape_context_histograms[j].clone()

                # 计算代价矩阵[num_sample,num_sample]
                cost_matrix = self.compute_cost_matrix(histograms1, histograms2)

                # print(f"the cost_matrix is {cost_matrix}")
                # 使用sinkhorn算法找到最优匹配
                _, total_cost = self.sinkhorn_matching(cost_matrix)

                # print(f"the total cost is: {total_cost}")
                # 将总代价存储到相似度矩阵中
                similarity_matrix[i, j] = total_cost
                similarity_matrix[j, i] = total_cost  # 对称矩阵
        max_cost=similarity_matrix.max()

        similarity = 1 - (similarity_matrix / max_cost)
        return similarity,similarity_matrix

    def sinkhorn_matching(self,cost_matrix, epsilon=0.01, n_iters=20):
        """ Sinkhorn-Knopp 近似匈牙利匹配（完全在 CUDA 计算） """
        N, M = cost_matrix.shape
        u = torch.zeros(N, device=cost_matrix.device)
        v = torch.zeros(M, device=cost_matrix.device)

        K = torch.exp(-cost_matrix / epsilon)  # 计算软分配矩阵

        for _ in range(n_iters):
            u = torch.logsumexp(-cost_matrix + v[None, :], dim=1)
            v = torch.logsumexp(-cost_matrix + u[:, None], dim=0)

        assignment = torch.argmax(K, dim=1)
        total_cost = cost_matrix[torch.arange(N, device=cost_matrix.device), assignment].sum()

        # print(f"the total_cost is {total_cost}")
        max_cost = total_cost.max()  # 假设已知最大代价


        return assignment, total_cost
    def find_top_k_similar_curves(self,similarity_matrix, shape_context_histograms, k=3):
        similarity_matrix = similarity_matrix.to(self.device)
        # 获取曲线的数量
        num_curves = len(shape_context_histograms)
        # 获取前三大相似度
        winner_1 = F.one_hot(torch.argmax(similarity_matrix, dim=0), num_classes=num_curves).T
        winner_2 = F.one_hot(torch.argmax(similarity_matrix - similarity_matrix * winner_1, dim=0),
                             num_classes=num_curves).T
        winner_3 = F.one_hot(torch.argmax(similarity_matrix - similarity_matrix * (winner_1 + winner_2), dim=0),
                             num_classes=num_curves).T
        # SP_adjmatrix = similarity_matrix * (1.0 * winner_1).to(self.device)
        SP_adjmatrix = similarity_matrix*(0.6 * winner_1 + 0.3* winner_2 + 0.1 * winner_3).to(self.device)
        # 按行归一化，使得每行总和为 1
        row_sum = SP_adjmatrix.sum(dim=0, keepdim=True)  # 计算每行的和
        SP_adjmatrix_normalized = SP_adjmatrix / row_sum  # 按行归一化
        # print(SP_adjmatrix_normalized)
        return SP_adjmatrix_normalized,similarity_matrix

