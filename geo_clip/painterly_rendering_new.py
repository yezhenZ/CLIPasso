import warnings

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

import os
import sys
import time
import traceback

import numpy as np
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torchvision import models, transforms
from tqdm.auto import tqdm, trange
import torchvision

import config
import sketch_utils as utils
from models.loss import Loss
from bezier_renderer import BezierRenderer
from models.painter_params import Painter, PainterOptimizer
from IPython.display import display, SVG
import model_rnngcn
from torch.utils.tensorboard import SummaryWriter
from bezier_renderer import BezierRenderer
from bezier import BezierCurve
from shape_context_2 import ShapeContext_2
from beziergen import SimpleBezierAE

def load_renderer(args, target_im=None, mask=None):
    renderer = Painter(num_strokes=args.num_paths, args=args,
                       num_segments=args.num_segments,
                       imsize=args.image_scale,
                       device=args.device,
                       target_im=target_im,
                       mask=mask)
    renderer = renderer.to(args.device)
    return renderer


def get_target(args):
    target = Image.open(args.target)
    if target.mode == "RGBA":
        # Create a white rgba background
        new_image = Image.new("RGBA", target.size, "WHITE")
        # Paste the image on the background.
        new_image.paste(target, (0, 0), target)
        target = new_image
    target = target.convert("RGB")
    # 利用 U2Net 模型生成图像的蒙版 mask，遮蔽后的图像 masked_im
    masked_im, mask = utils.get_mask_u2net(args, target)
    if args.mask_object:
        target = masked_im
    if args.fix_scale:
        target = utils.fix_image_scale(target)
    # 图像缩放调整
    transforms_ = []
    if target.size[0] != target.size[1]:
        transforms_.append(transforms.Resize(
            (args.image_scale, args.image_scale), interpolation=PIL.Image.BICUBIC))
    else:
        transforms_.append(transforms.Resize(
            args.image_scale, interpolation=PIL.Image.BICUBIC))
        transforms_.append(transforms.CenterCrop(args.image_scale))
    transforms_.append(transforms.ToTensor())
    data_transforms = transforms.Compose(transforms_)
    target_ = data_transforms(target).unsqueeze(0).to(args.device)
    return target_, mask


def init_writer():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    runs_dir = os.path.join(base_dir, "paint_render")
    save_path = os.path.join(base_dir, "Sp_matrix")
    img_dir = os.path.join(base_dir, "render_imgs")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for path in [runs_dir, save_path]:
        os.makedirs(path, exist_ok=True)

    writer = SummaryWriter(runs_dir)
    return writer, runs_dir, save_path, img_dir


def main(args):
    mseloss = torch.nn.MSELoss()
    loss_func = Loss(args)
    inputs, mask = get_target(args)
    utils.log_input(args.use_wandb, 0, inputs, args.output_dir)
    renderer = load_renderer(args, inputs, mask)
    writer, runs_dir, save_path, img_dir = init_writer()

    # 定义形状上下文的参数
    nbins_r = 6
    nbins_theta = 12
    r_inner = 0.125  # 内半径
    r_outer =1.5  # 外半径
    wlog = False  # 是否使用对数尺度
    num_points = 20  # 采样点个数

    # 创建 ShapeContext 对象
    shape_context_2 = ShapeContext_2(nbins_r=nbins_r, nbins_theta=nbins_theta, r_inner=r_inner, r_outer=r_outer,
                                     wlog=wlog)

    # 创建bezier曲线参数化对象
    bezier_render = BezierRenderer()
    bezier = BezierCurve(num_points=100)
    # 创建模型
    Pretrained_model = SimpleBezierAE().to(args.device)
    checkpoint = torch.load('best_model.pth')  # 加载文件
    Pretrained_model.load_state_dict(checkpoint['state_dict'])  # 加载模型参数
    ctrlpt_head = Pretrained_model.ctrlpt_head
    for p in ctrlpt_head.parameters():
        p.requires_grad = False
    GCNmodel = model_rnngcn.GCN(input_dim=128, output_dim=4, ctrlpt_head=ctrlpt_head).to(args.device)  # 4 是每条bezier曲线控制点的数量

    optimizer = PainterOptimizer(args, list(GCNmodel.parameters()), renderer)

    # 初始化训练
    renderer.set_random_noise(0)
    img = renderer.init_image(stage=0)
    optimizer.init_optimizers()
    counter = 0
    configs_to_save = {"loss_eval": []}
    best_loss, best_fc_loss = 100, 100
    best_iter, best_iter_fc = 0, 0
    epoch_gcn_start=100
    min_delta = 1e-5
    terminate = False

    # not using tdqm for jupyter demo
    if args.display:

        epoch_range = range(args.num_iter)
    else:
        epoch_range = tqdm(range(args.num_iter))
    for epoch in epoch_range:
        if not args.display:
            epoch_range.refresh()
        renderer.set_random_noise(epoch)
        if args.lr_scheduler:
            optimizer.update_lr(counter)
        start = time.time()
        optimizer.zero_grad_()

        # 获取控制点
        for i, path in enumerate(renderer.shapes):
            renderer.control_points_set[i] = path.points
        points = torch.stack(renderer.control_points_set, dim=0)
        nom_points=points/224.0
        sampled_points_list = bezier.batch_sample(control_points=nom_points)
        # print(sampled_points_list.shape)
        Pretrained_points ,feature= Pretrained_model(sampled_points_list)
        # if epoch%100==0:
        #     print(f"the Pretrained_points is {Pretrained_points*224}")
        #     print(f"the points is {points}")

        # 计算形状上下文
        sampled_points_list = bezier.batch_sample(control_points=Pretrained_points*224)

        #  #【curves,samples,直方图】
        shape_context_histograms,histograms = shape_context_2.compute(sampled_points_list)
        shape_context_matrix ,_= shape_context_2.compute_similarity(shape_context_histograms)

        SP_adjmatrix_normalized,SP_adjmatrix = shape_context_2.find_top_k_similar_curves(shape_context_matrix, shape_context_histograms, k=3)
        # print(feature)
        #gcn
        new_points = GCNmodel(feature, SP_adjmatrix_normalized).view(-1, 4, 2)
        # print(new_points)
        if epoch <=200:
            # sketch绘制
            sketches = utils.render_img_rgb_from_renderer(Pretrained_points, renderer, sampled_points_list,epoch, img_dir).to(args.device)
        else :
            sketches = utils.render_img_rgb_from_renderer(new_points, renderer, sampled_points_list,epoch, img_dir).to(args.device)

        losses_dict = loss_func(sketches, inputs.detach(), renderer.get_color_parameters(), renderer, counter,
                                optimizer)

        cliploss = sum(list(losses_dict.values()))
        loss = cliploss
        loss.backward()

        # 检查损失值
        optimizer.step_()
        if epoch % args.save_interval == 0:
            # torch.save(GCNmodel.state_dict(),"gcnmodel.pth")
            # if epoch==1600:
            #     # 转换为 numpy 数组
            #     Pretrained_points_1600 = Pretrained_points.clone().detach().cpu().numpy()

                # # 保存为 .npz 文件
                # np.save('control_points_epoch1600.npy',Pretrained_points_1600)
            # 打印网格的形状,chw
            utils.save_cosine_similarity_heatmap(SP_adjmatrix, save_path, epoch, "SP_adjmatrix")
           
            # 保存为 .pt 文件（PyTorch 张量格式）
            # torch.save(feature, f"{args.output_dir}/feature.pt")
            # 假设 model 是你的模型

            utils.plot_batch(inputs, sketches, f"{args.output_dir}/jpg_logs", counter,
                             use_wandb=args.use_wandb, title=f"iter{epoch}.jpg")
            renderer.save_svg(
                f"{args.output_dir}/svg_logs", f"svg_iter{epoch}")
        if epoch % args.eval_interval == 0:

            losses_dict_eval = loss_func(sketches, inputs, renderer.get_color_parameters(
            ), renderer.get_points_parans(), counter, optimizer, mode="eval")
            loss_eval = sum(list(losses_dict_eval.values()))
            configs_to_save["loss_eval"].append(loss_eval.item())
            for k in losses_dict_eval.keys():
                if k not in configs_to_save.keys():
                    configs_to_save[k] = []
                configs_to_save[k].append(losses_dict_eval[k].item())
            if args.clip_fc_loss_weight:
                if losses_dict_eval["fc"].item() < best_fc_loss:
                    best_fc_loss = losses_dict_eval["fc"].item(
                    ) / args.clip_fc_loss_weight
                    best_iter_fc = epoch

            print(
                f"eval iter[{epoch}/{args.num_iter}] loss[{loss.item()}] time[{time.time() - start}]")
            # 保存loss情况
            writer.add_scalar("loss_total.", loss.item(), global_step=epoch)

        #     print(
        #                 f"eval iter[{epoch}/{args.num_iter}] loss_total[{loss.item()}] time[{time.time() - start}],")
            # 保存loss情况
            writer.add_scalar("loss_total.", loss.item(), global_step=epoch)


            cur_delta = loss_eval.item() - best_loss
            if abs(cur_delta) > min_delta:
                if cur_delta < 0:
                    best_loss = loss_eval.item()
                    best_iter = epoch
                    terminate = False
                    utils.plot_batch(
                        inputs, sketches, args.output_dir, counter, use_wandb=args.use_wandb, title="best_iter.jpg")
                    renderer.save_svg(args.output_dir, "best_iter")

            if args.use_wandb:
                wandb.run.summary["best_loss"] = best_loss
                wandb.run.summary["best_loss_fc"] = best_fc_loss
                wandb_dict = {"delta": cur_delta,
                              "loss_eval": loss_eval.item()}
                for k in losses_dict_eval.keys():
                    wandb_dict[k + "_eval"] = losses_dict_eval[k].item()
                wandb.log(wandb_dict, step=counter)

        if abs(cur_delta) <= min_delta:
            if terminate:
                break
            terminate = True

    if counter == 0 and args.attention_init:
        utils.plot_atten(renderer.get_attn(), renderer.get_thresh(), inputs, renderer.get_inds(),
                         args.use_wandb, "{}/{}.jpg".format(
                args.output_dir, "attention_map"),
                         args.saliency_model, args.display_logs)

    if args.use_wandb:
        wandb_dict = {"loss": loss.item(), "lr": optimizer.get_lr()}
        for k in losses_dict.keys():
            wandb_dict[k] = losses_dict[k].item()
        wandb.log(wandb_dict, step=counter)

    counter += 1

    path_svg = os.path.join(args.output_dir, "best_iter.svg")
    utils.log_sketch_summary_final(
        path_svg, args.use_wandb, args.device, best_iter, best_loss, "best total")

    return configs_to_save


if __name__ == "__main__":
    args = config.parse_arguments()
    final_config = vars(args)
    try:
        configs_to_save = main(args)
    except BaseException as err:
        print(f"Unexpected error occurred:\n {err}")
        print(traceback.format_exc())
        sys.exit(1)
    for k in configs_to_save.keys():
        final_config[k] = configs_to_save[k]
    np.save(f"{args.output_dir}/config.npy", final_config)
    if args.use_wandb:
        wandb.finish()


