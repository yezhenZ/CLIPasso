'''
生成在画布内的控制点，对每条画出的曲线进行掩码，给u-net或者cnn,提取特征（训练全连接层），生成图片（转化为控制点）或者控制点

'''
import torch
from torch.utils.data import Dataset, DataLoader,random_split
import torch.nn as nn
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import os
import random
# from pytorch_batch_sinkhorn import SinkhornLoss
# from torchvision.models import vit_b_16, ViT_B_16_Weights
from bezier import BezierCurve
# 生成在画布内的控制点,三阶bezier曲线，4个控制点，01之间
from beziergen import SimpleBezierAE
from bezier_renderer import BezierRenderer
class BezierTrainer:
    def __init__(self, model,writer, device=None):
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.bezier_curve = BezierCurve(num_points=100)
        self.writer=writer
        self.bezier_renderer=BezierRenderer(224, 224)


    #train loss
    def train_epoch(self, optimizer,criterion,epoch):
        self.model.train()
        total_loss = 0.0
        total_loss_points = 0.0
        total_loss_imgs = 0.0
        num = random.randint(0, 5)  # 生成 1 到 16 之间的随机整数
        dataset = torch.rand(9600, 4, 2, device='cuda')
        train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
        for idx, inputs in enumerate(train_loader):  #input 是batchsize ，4,2
            inputs = inputs.to(self.device)
            #绘制曲线并每条曲线采样100个点
            sampled_points_list = self.bezier_curve.batch_sample(control_points=inputs)

            optimizer.zero_grad()

            outputs,_ = self.model(sampled_points_list)
            # 单个曲线绘制
            img_in = self.bezier_renderer.render_beziers(inputs *224)
            # render_bezier就是img_in = img_in.permute(0, 3, 1, 2)
            img_in = img_in.permute(0, 3, 1, 2)
            img_in = img_in.repeat(1, 3, 1, 1)
            img_out = self.bezier_renderer.render_beziers(outputs * 224)
            img_out = img_out.permute(0, 3, 1, 2)
            img_out = img_out.repeat(1, 3, 1, 1)
            # print(outputs.shape)

            if epoch % 5 == 0 and idx==num:
                self.writer.add_image(f'{epoch}img_out', img_out[num])
                self.writer.add_image(f'{epoch}img_out{num+1}', img_out[num+1])
                self.writer.add_image(f'{epoch}img_rare', img_in[num])
                self.writer.add_image(f'{epoch}img_rare{num+1}', img_in[num+1])

            target=inputs.clone().detach()
            # target_img=img_in.clone().detach()
            loss_points = criterion(outputs, target)
            # loss_imgs= criterion(img_out, target_img)
            loss = loss_points
            loss.backward()
            optimizer.step()

            # 累加每个batch的损失
            total_loss += loss.item()
            # total_loss_points+= loss_points.item()
            # total_loss_imgs+= loss_imgs.item()


        # return total_loss / len(train_loader),total_loss_points/ len(train_loader),total_loss_imgs/ len(train_loader)
        return total_loss / len(train_loader)
    def fit(self ,epochs=100,
                early_stopping_patience=5, checkpoint_path=None):

            optimizer = torch.optim.AdamW(self.model.parameters(),lr=1e-4)
            # 连续 3 个 epoch 不下降就降低学习率
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=20)
            criterion = nn.SmoothL1Loss()  # 对异常值比MSE鲁棒
            mse_loss = nn.MSELoss()
            # sinkhorn_loss=SinkhornLoss()
            best_loss = float('inf')
            patience_counter = 0

            for epoch in range(epochs):
                # train_loss ,loss_points,loss_imgs= self.train_epoch( optimizer, sinkhorn_loss,criterion,epoch)
                train_loss= self.train_epoch( optimizer,criterion,epoch)

                # 打印每层梯度的范数
                # if epoch %5==0:
                #     for name, param in self.model.named_parameters():
                #         if param.grad is not None:
                #             print(f"{name}: grad norm = {param.grad.norm().item():.4f}")

                scheduler.step(train_loss)
                print(f"Epoch {epoch + 1}/{epochs} | "
                      f"Train Loss: {train_loss:.4f} | "
                      # f"loss_points: {loss_points:.4f} | "
                      # f"loss_imgs: {loss_imgs:.4f} | "
                      f"LR: {optimizer.param_groups[0]['lr']:.2e}")
                # self.writer.add_scalar('loss_points', loss_points, epoch)
                # self.writer.add_scalar('loss_imgs', loss_imgs, epoch)
                self.writer.add_scalar('Train Loss:', train_loss, epoch)
                if epoch%10==0:
                    self.model.save(checkpoint_path)
                # # Early stopping
                # if val_loss < best_loss:
                #     best_loss = val_loss
                #     patience_counter = 0
                #     if checkpoint_path:
                #
                # else:
                #     patience_counter += 1
                #     if patience_counter >= early_stopping_patience:
                #         print(f"Early stopping at epoch {epoch + 1}")
                #         break

if __name__ == '__main__':
    #数据前置处理
    #一些参数
    base_dir = os.path.dirname(os.path.abspath(__file__))
    runs_dir = os.path.join(base_dir, "runs")
    if not os.path.exists(runs_dir):
        os.makedirs(runs_dir)
    writer = SummaryWriter(runs_dir)
    data=np.load("control_points1500.npy")
    tensor = torch.from_numpy(data)
    bezier_renderer = BezierRenderer(224, 224)
    torch.cuda.empty_cache()  # 清空缓存
    #模型
    model = SimpleBezierAE()
    trainer = BezierTrainer(model,writer)
    # 训练
    trainer.fit(
        epochs=100,
        checkpoint_path="best_model_alayer.pth"
    )

    # model.eval()
    # checkpoint = torch.load('best_model.pth')  # 加载文件
    # model.load_state_dict(checkpoint['state_dict'])  # 加载模型参数
    # test_input=torch.rand(96, 4, 2, device='cuda')
    # with torch.no_grad():  # 禁用梯度计算
    #     test_output = model(tensor)  # 运行测试
    #     bezier_masked = bezier_renderer.render_img_raw(test_output * 224)
    #     print(test_output)
    #     print(tensor)
    #     # img = bezier_masked.permute(1, 2, 0)
    #     # img = img.squeeze()
    #     # 将张量转换为 NumPy 数组
    #     image_np = bezier_masked.detach().cpu().numpy()
    #
    #     # 使用 matplotlib 绘制图像
    #     plt.imshow(image_np)  # 使用灰度颜色映射
    #     plt.title("64x64 Grayscale Image")  # 添加标题
    #     plt.axis('off')  # 关闭坐标轴
    #     plt.savefig(f'bezier_masked.png')  # 显示图像
    #     print(test_output.shape)
    #
    #     image_rare = bezier_renderer.render_img_raw(tensor * 224)
    #
    #     image_np = image_rare.detach().cpu().numpy()
    #
    #     # 使用 matplotlib 绘制图像
    #     plt.imshow(image_np)  # 使用灰度颜色映射
    #     plt.title("64x64 Grayscale Image")  # 添加标题
    #     plt.axis('off')  # 关闭坐标轴
    #     plt.savefig(f'img_rare.png')  # 显示图像
    #     print(test_output.shape)





    # #原本是batchsize，c ，h，w，现在是cbhw

    #
    # batch = next(iter(train_loader))  # 使用iter()创建迭代器
    # bezier_renderer = BezierRenderer(224, 224)
    # print(batch.shape)
    # bezier_masked = bezier_renderer.mask_img(batch* 224)
    # imgs=bezier_renderer.render_beziers(batch* 224)
    # bezier_masked=bezier_masked.permute(1, 0, 2, 3)
    # writer.add_image(f'img_out', bezier_masked[5])
    #
    # writer.close()
    #
    # print(bezier_masked.shape)
    # for i in range(imgs.shape[0]):
    #     img = imgs[i]
    #     # 将张量转换为 NumPy 数组
    #     image_np = img.detach().cpu().numpy()
    #
    #     # 使用 matplotlib 绘制图像
    #     plt.imshow(image_np,cmap='gray')  # 使用灰度颜色映射
    #     plt.title("64x64 Grayscale Image")  # 添加标题
    #     plt.axis('off')  # 关闭坐标轴
    #     plt.savefig(f'img{i}.png')  # 显示图像

    # for i in range(bezier_masked.shape[0]):
    #     img = bezier_masked[i].permute(1, 2, 0)
    #     img = img.squeeze()
    #     print(img.shape)
    #     # 将张量转换为 NumPy 数组
    #     image_np = img.detach().cpu().numpy()
    #
    #     # 使用 matplotlib 绘制图像
    #     plt.imshow(image_np,cmap='gray')  # 使用灰度颜色映射
    #     plt.title("64x64 Grayscale Image")  # 添加标题
    #     plt.axis('off')  # 关闭坐标轴
    #     plt.savefig(f'bezier_masked{i}.png')  # 显示图像


    #
    # writer.flush()  # ← 关键！强制立即写入磁盘


    #
    # print(img_rare.shape)
    # # 去除批次和通道维度
    # image = bezier_masked.squeeze()  # 形状变为 [16,64, 64]
    # for i in range(image.shape[0]):
    #     # 将张量转换为 NumPy 数组
    #     image_np = image[i].detach().cpu().numpy()
    #     # 使用 matplotlib 绘制图像
    #     plt.imshow(image_np, cmap='gray')  # 使用灰度颜色映射
    #     plt.title("64x64 Grayscale Image")  # 添加标题
    #     plt.axis('off')  # 关闭坐标轴
    #     plt.savefig(f'test{i}.png')  # 显示图像
    #     plt.close()
    #
    #绘制原始图像
    # print(img_rare.shape)
    # image = img_rare.squeeze()  # 形状变为 [224, 224]


    # 将张量转换为 NumPy 数组
    # image_np = img_rare_1.detach().cpu().numpy()

    # 使用 matplotlib 绘制图像
    # plt.imshow(image_np)  # 使用灰度颜色映射
    # plt.title("64x64 Grayscale Image")  # 添加标题
    # plt.axis('off')  # 关闭坐标轴
    # plt.savefig('test.png')  # 显示图像