import torch
import numpy as np
import random
import cv2

from gaussian_renderer import render
from scene import Scene
from utils.image_utils import psnr
from utils.loss_utils import ssim

import torch.nn.functional as F
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
import os


def load_mask(path):

    mask = cv2.imread(path, 0)

    if mask is None:
        # 如果找不到 Mask，打印警告并返回一个全黑的掩码（假设不遮挡任何东西）
        # 注意：这里的尺寸 1024 只是占位，实际运行中如果报错尺寸不符，
        # 建议在 evaluate 传入 image 的尺寸来初始化这个 zeros
        print(f"Warning: Mask not found at {path}")
        return torch.zeros((1, 1))  # 或者根据你的逻辑返回 None
    mask = (mask > 128).astype(np.float32)
    mask = torch.from_numpy(mask)
    return mask


def warp_with_depth(
    ref_image,
    ref_depth,
    ref_cam,
    target_cam
):
    ref_image = ref_image.float()
    ref_depth = ref_depth.float()

    dtype = ref_image.dtype
    device = ref_image.device

    if ref_depth.dim() == 3:
        ref_depth = ref_depth.squeeze()

    H, W = ref_depth.shape

    ys, xs = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )

    xs = xs.float()
    ys = ys.float()

    fovx = torch.tensor(ref_cam.FoVx, device=device)
    fovy = torch.tensor(ref_cam.FoVy, device=device)

    tanx = torch.tan(fovx / 2)
    tany = torch.tan(fovy / 2)

    x = (xs / W * 2 - 1) * tanx
    y = (ys / H * 2 - 1) * tany

    z = ref_depth

    pts_cam = torch.stack(
        [x*z, y*z, z],
        dim=-1
    )

    R = torch.tensor(ref_cam.R, device=device)
    T = torch.tensor(ref_cam.T, device=device)

    pts_world = (
        R.T @ (pts_cam.reshape(-1,3).T - T[:,None])
    ).T

    R_t = torch.tensor(target_cam.R, device=device)
    T_t = torch.tensor(target_cam.T, device=device)

    pts_target = (
        R_t @ pts_world.T + T_t[:,None]
    )

    x_proj = pts_target[0] / pts_target[2]
    y_proj = pts_target[1] / pts_target[2]

    fovx_t = torch.tensor(target_cam.FoVx, device=device)
    fovy_t = torch.tensor(target_cam.FoVy, device=device)

    grid_x = x_proj / torch.tan(fovx_t/2)
    grid_y = y_proj / torch.tan(fovy_t/2)

    grid = torch.stack(
        [grid_x, grid_y],
        dim=-1
    )

    grid = grid.reshape(H, W, 2)
    grid = grid.unsqueeze(0)

    grid = grid.float()  # ⭐ 最关键一行

    warped = F.grid_sample(
        ref_image.unsqueeze(0),
        grid,
        align_corners=True
    )

    return warped.squeeze(0)


def evaluate(scene, pipe, dataset):

    cameras = scene.getTrainCameras()
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    psnr_vals = []
    ssim_vals = []

    for cam in cameras:

        name = cam.image_name

        overlap_entries = scene.overlap_dict.get(
            str(name), []
        )

        if len(overlap_entries) == 0:
            continue

        neighbor_name = overlap_entries[0][0]

        neighbor_cam = scene.getTrainCamerasFromName(
            neighbor_name
        )

        if neighbor_cam is None:
            continue

        render_pkg = render(
            cam,
            scene.gaussians,
            pipe,
            background
        )

        image = render_pkg["render"]
        depth = render_pkg["depth"]

        ref_gt = neighbor_cam.original_image.cuda()

        warped = warp_with_depth(
            image,
            depth,
            cam,
            neighbor_cam
        )

        file_id = os.path.splitext(name)[0]
        mask_path = os.path.join(dataset.source_path, "masks", file_id + ".png")

        mask = load_mask(mask_path).cuda()

        # 1. 确保 mask 是 [1, H, W] 格式
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        # 2. 检查并缩放 mask 到 warped 的尺寸
        if mask.shape[-2:] != warped.shape[-2:]:
            # 使用 nearest (最近邻) 插值，防止掩码边缘产生模糊的中间值
            mask = F.interpolate(
                mask.unsqueeze(0),
                size=warped.shape[-2:],
                mode='nearest'
            ).squeeze(0)

        # 3. 检查并缩放 ref_gt (参考真值图) 到 warped 的尺寸
        # 因为跨视角投影后的尺寸可能与原图不一致
        if ref_gt.shape[-2:] != warped.shape[-2:]:
            ref_gt = F.interpolate(
                ref_gt.unsqueeze(0),
                size=warped.shape[-2:],
                mode='bilinear',
                align_corners=True
            ).squeeze(0)

        warped = warped * mask
        ref_gt = ref_gt * mask

        valid_idx = mask > 0.5


        psnr_val = psnr(
            warped,
            ref_gt
        )

        ssim_val = ssim(
            warped.unsqueeze(0),
            ref_gt.unsqueeze(0),
        )

        if valid_idx.any():
            # 只取有效区域的像素点
            mse = torch.mean((warped[valid_idx.expand_as(warped)] -
                              ref_gt[valid_idx.expand_as(ref_gt)]) ** 2)
            psnr_val_correct = 20 * torch.log10(1.0 / torch.sqrt(mse))
            psnr_vals.append(psnr_val_correct.item())
        else:
            print(f"Skip {name}: No valid pixels after masking.")
        ssim_vals.append(ssim_val.mean().item())

        print(
            name,
            psnr_val.mean().item(),
            ssim_val.mean().item()
        )

    print("==== Final Result ====")

    print(
        "PSNR:",
        np.mean(psnr_vals)
    )

    print(
        "SSIM:",
        np.mean(ssim_vals)
    )


if __name__ == "__main__":

    parser = ArgumentParser()

    lp = ModelParams(parser)
    pp = PipelineParams(parser)

    args = parser.parse_args()

    dataset = lp.extract(args)
    pipe = pp.extract(args)

    from gaussian_renderer import GaussianModel

    gaussians = GaussianModel(
        dataset.sh_degree
    )

    scene = Scene(
        dataset,
        gaussians,
        shuffle=False
    )

    evaluate(scene, pipe, dataset)