import torch.nn.functional as F
import torch
print("Reverse projection using depth information")

def warp_image_to_view(ref_image, ref_cam, target_cam, gaussians, pipe, depth_map):
    """
    将 ref_cam 的图像（渲染结果）ref_image 投影到 target_cam 视角下

    ref_image:    [3, H, W] 渲染图像（参考视角）
    ref_cam:      Camera 对象（参考视角）
    target_cam:   Camera 对象（目标视角）
    z:            默认射线深度（可以设为1.0）

    return:
        warped_image: [3, H, W] 在 target_cam 视角下的参考视角图像重投影结果
    """

    device = ref_image.device
    H, W = target_cam.image_height, target_cam.image_width

    if depth_map.shape != (H, W):
        depth_map = F.interpolate(
            depth_map.unsqueeze(0).unsqueeze(0),  # [1,1,H,W]
            size=(H, W),
            mode='bilinear',
            align_corners=True
        ).squeeze()  # [H, W]

    depth_map = depth_map.to(device)
    # 1. 生成归一化像素坐标（NDC） [-1, 1]
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing='ij'
    )
    xy_ndc = torch.stack([x, y], dim=-1)  # [H, W, 2]

    # 2. 计算 target 相机的方向向量（ray directions）
    fov_x = torch.tensor(target_cam.FoVx, device=device)
    fov_y = torch.tensor(target_cam.FoVy, device=device)
    tan_fovx = torch.tan(fov_x / 2)
    tan_fovy = torch.tan(fov_y / 2)

    # NDC -> 相机坐标系
    x_cam = xy_ndc[..., 0] * tan_fovx
    y_cam = xy_ndc[..., 1] * tan_fovy
    z_cam = depth_map
    dirs = torch.stack([x_cam, y_cam, z_cam], dim=-1)  # [H, W, 3]


    dirs = dirs.reshape(-1, 3).transpose(0, 1)  # [3, N]
    R = torch.tensor(target_cam.R, dtype=torch.float32, device=device)
    T = torch.tensor(target_cam.T, dtype=torch.float32, device=device).view(3, 1)
    rays_world = R @ dirs + T  # [3, N]
    points_world = rays_world.T  # [N, 3]


    R_ref = torch.tensor(ref_cam.R, dtype=torch.float32, device=device)
    T_ref = torch.tensor(ref_cam.T, dtype=torch.float32, device=device).view(3, 1)
    pts_cam = R_ref @ (points_world.T - T_ref)  # [3, N]


    x_proj = pts_cam[0] / pts_cam[2]
    y_proj = pts_cam[1] / pts_cam[2]

    fovx_ref = torch.tensor(ref_cam.FoVx, device=device)
    fovy_ref = torch.tensor(ref_cam.FoVy, device=device)
    tan_ref_fovx = torch.tan(fovx_ref / 2)
    tan_ref_fovy = torch.tan(fovy_ref / 2)

    u = x_proj / tan_ref_fovx
    v = y_proj / tan_ref_fovy


    grid_x = u.view(H, W)
    grid_y = v.view(H, W)
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # [1, H, W, 2]


    ref_image = ref_image.unsqueeze(0)  # [1, 3, H, W]
    warped = F.grid_sample(ref_image, grid, mode='bilinear', padding_mode='border', align_corners=True)
    return warped.squeeze(0)  # [3, H, W]

