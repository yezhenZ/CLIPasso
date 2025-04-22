import numpy as np
import torch
from math import comb as n_over_k  # Comb(n, k) 计算组合数
from matplotlib import pyplot as plt
import math
from bezier_renderer import BezierRenderer
from shape_context_2 import ShapeContext_2
import cv2
import os

class BezierCurve:
    def __init__(self, num_points=100, num_degree=3,device='cuda'):
        """
        :param num_points: 每条贝塞尔曲线采样点的数量
        :param num_degree: 贝塞尔曲线的阶数，3阶贝塞尔曲线需要4个控制点
        """
        self.num_points = num_points  # 曲线的采样点数
        self.num_degree = num_degree  # 3阶贝塞尔曲线，即4个控制点
        self.device = device  # 设备类型
        # 预计算贝塞尔系数
        self.t_values = torch.linspace(0, 1, num_points, device=device)
        self.bezier_coeff = self._get_bezier_coefficients()

    def comb_torch(self,n, k):
        """使用 torch.lgamma 计算组合数"""
        return torch.exp(torch.lgamma(n + 1) - (torch.lgamma(k + 1) + torch.lgamma(n - k + 1)))

    def _get_bezier_coefficients(self):
        """预计算所有t值的贝塞尔系数"""
        n = self.num_degree
        k = torch.arange(n + 1, device=self.device)
        n = torch.tensor(n, dtype=torch.float32, device=k.device)  # 确保 n 也是 Tensor

        # 使用广播加速计算
        t = self.t_values.view(-1, 1)
        coeff = self.comb_torch(n, k) * (t ** k) * ((1 - t) ** (n - k))
        return coeff

    def batch_sample(self, control_points):
        """
        批量生成贝塞尔曲线
        :param control_points: (batch_size, num_degree+1, 2)
        :return: (batch_size, num_points, 2)
        """
        # print(control_points.shape)
        # print(self.bezier_coeff.shape)
        # 使用爱因斯坦求和约定加速计算
        return torch.einsum('nd,bdc->bnc', self.bezier_coeff, control_points)



    # def plot_curve_and_control_points(self, control_points, sampled_points):
    #     """
    #     绘制贝塞尔曲线和控制点
    #     :param control_points: 控制点
    #     :param sampled_points: 采样的曲线点
    #     """
    #     # 生成光滑的贝塞尔曲线
    #     t_values = np.linspace(0, 1, 1000)  # 用更多点来生成光滑曲线
    #     smooth_curve = self.bezier_curve(t_values, control_points)
    #     # 绘制贝塞尔曲线
    #     plt.plot(smooth_curve[:, 0], smooth_curve[:, 1], label="Bezier Curve", color='blue')
    #
    #     # 绘制控制点
    #     plt.scatter(control_points[:, 0], control_points[:, 1], color='red', label="Control Points")
    #     plt.plot(control_points[:, 0], control_points[:, 1], 'r--', label="Control Polygon")
    #
    #     # 绘制采样点
    #     plt.scatter(sampled_points[:, 0], sampled_points[:, 1], color='green', label="Sampled Points", s=50)
    #
    #     plt.legend()
    #     plt.title("Bezier Curve and Control Points")
    #     plt.savefig("bezier_curve.png")
        # plt.show()


if __name__ == '__main__':
    # 定义形状上下文的参数
    nbins_r = 7
    nbins_theta = 12
    r_inner = 0.125  # 内半径
    r_outer = 2.5  # 外半径
    wlog = False  # 是否使用对数尺度

    # 创建 ShapeContext 对象
    ShapeContext_batch = ShapeContext_batch(nbins_r=nbins_r, nbins_theta=nbins_theta, r_inner=r_inner, r_outer=r_outer, wlog=wlog)
    shape_context_2 = ShapeContext_2(nbins_r=nbins_r, nbins_theta=nbins_theta, r_inner=r_inner, r_outer=r_outer, wlog=wlog)

    bezier_render = BezierRenderer()
    # data = np.load('control_points1500.npy')
    #绘制原始图片
    data = torch.from_numpy(np.load('control_points1500.npy')).to("cuda")
    data.requires_grad_(True)  # 在原地设置 requires_grad 为 True
    sampled_points_list=[]
    shape_context_histograms = []

    # for num,control_point in enumerate(data):
    #     #每条曲线采样五个点
    #     bezier=BezierCurve(num_points=15)
    #     #绘制曲线
    #     sampled_points=bezier.sample_from_curve(control_point)
    #     sampled_points_list.append(sampled_points)
    #
    # sampled_points_list_test=torch.stack(sampled_points_list)

    bezier = BezierCurve(num_points=100)
    sampled_points_list_test=bezier.batch_sample(control_points=data)
    # print(sampled_points_list_test)
    shape_context_histogram = shape_context_2.compute(sampled_points_list_test)
    # print(shape_context_histogram)
    # 指定保存图像的文件夹
    output_folder = "bezier_curves_samples"
    os.makedirs(output_folder, exist_ok=True)  # 创建文件夹（如果不存在）
    # # shape_context_2.plot_and_save_histogram(histogram,"bezier_curves_samples",1)
    shape_context_matrix = shape_context_2.compute_similarity(shape_context_histogram)
    # print(shape_context_matrix)

    # 颜色列表
    colors = plt.cm.tab20.colors  # 使用 Matplotlib 的颜色映射
    plt.figure(figsize=(10, 8))  # 创建一个新的图像
    for num, sampled_points in enumerate(sampled_points_list_test):
        # sampled_points 是形状为 (10, 2) 的数组，其中包含 10 个 (x, y) 坐标
        sampled_x, sampled_y = (sampled_points[:, 0]*224).detach().cpu().numpy(), (sampled_points[:, 1]*224).detach().cpu().numpy()  # 提取 x 和 y 坐标
        # 在曲线的中间点旁边添加标签
        mid_x, mid_y = sampled_x[len(sampled_x) // 2], sampled_y[len(sampled_y) // 2]
        plt.text(mid_x, mid_y, f'Curve {num + 1}', color='black', fontsize=10, verticalalignment='bottom',
                 horizontalalignment='right')

        # 绘制每条曲线的采样点
        plt.plot(sampled_x, sampled_y, 'o-', color=colors[num % len(colors)],label=f'Curve {num + 1}')  # 使用圆点和线连接

    # 设置图形标题和标签
    plt.title('Sampled Points for Bezier Curves')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()
    plt.savefig('sampled_points.png')