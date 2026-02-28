import torch
import torch.nn as nn
import torch.nn.functional as F
import gc

from functools import partial

from .utils import split_feature, merge_splits, split_feature_1d, merge_splits_1d


def multihead_split_window_attention_urope(q, k, v,
        num_splits=1, with_shift=False, h=None, w=None, attn_mask=None, intrinsics=None, pose=None, num_heads=4, min_depth=2.0, max_depth=20.0, is_self_attn=False):
    # ref: https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
    # q, k, v: [B, L, C]
    assert q.dim() == k.dim() == v.dim() == 3
    
    if is_self_attn:
        num_heads = 1
    
    assert h is not None and w is not None
    assert q.size(1) == h * w

    b, _, c = q.size()

    b_new = b * num_splits * num_splits

    window_size_h = h // num_splits
    window_size_w = w // num_splits

    c_head = c // num_heads
    assert c_head * num_heads == c

    apply_fn_q, apply_fn_kv, apply_fn_o = compute_urope(intrinsics, pose, h, w, head_dim=c_head, num_heads=num_heads, min_depth=min_depth, max_depth=max_depth, device=q.device)
    
    q = q.reshape(b, h*w, num_heads, c_head).permute(0, 2, 1, 3) # [B, num_heads, h*w, c_head]
    k = k.reshape(b, h*w, num_heads, c_head).permute(0, 2, 1, 3) 
    v = v.reshape(b, h*w, num_heads, c_head).permute(0, 2, 1, 3) 
    
    q = apply_fn_q(q)
    k = apply_fn_kv(k)
    v = apply_fn_o(v)
    
    q = q.permute(1, 0, 2, 3) # [num_heads, B, h*w, c_head]
    k = k.permute(1, 0, 2, 3) # [num_heads, B, h*w, c_head]
    v = v.permute(1, 0, 2, 3) # [num_heads, B, h*w, c_head]
    
    q = q.reshape(b*num_heads, h, w, c_head)  # [B*num_heads, H, W, c_head]
    k = k.reshape(b*num_heads, h, w, c_head)  # [B*num_heads, H, W, c_head]
    v = v.reshape(b*num_heads, h, w, c_head)  # [B*num_heads, H, W, c_head]

    scale_factor = c_head ** 0.5

    if with_shift:
        assert attn_mask is not None  # compute once
        shift_size_h = window_size_h // 2
        shift_size_w = window_size_w // 2

        q = torch.roll(q, shifts=(-shift_size_h, -shift_size_w), dims=(1, 2))
        k = torch.roll(k, shifts=(-shift_size_h, -shift_size_w), dims=(1, 2))
        v = torch.roll(v, shifts=(-shift_size_h, -shift_size_w), dims=(1, 2))

    q = split_feature(q, num_splits=num_splits, channel_last=True)  # [B*K*K*num_heads, H/K, W/K, C_head]
    k = split_feature(k, num_splits=num_splits, channel_last=True)
    v = split_feature(v, num_splits=num_splits, channel_last=True)
    
    q = q.reshape(num_heads, b_new, -1, c_head) # [num_heads, B*K*K, H/K*W/K, C_head]
    k = k.reshape(num_heads, b_new, -1, c_head) # [num_heads, B*K*K, H/K*W/K, C_head]
    v = v.reshape(num_heads, b_new, -1, c_head) # [num_heads, B*K*K, H/K*W/K, C_head]

    # 分批处理以减少显存占用
    batch_chunk_size = max(1, b_new)  # 每次处理 1/4 的 batch，可根据显存调整
    out = []
    attn_mask = attn_mask.repeat(b, 1, 1)
    for i in range(num_heads):
        out_head_chunks = []
        for chunk_start in range(0, b_new, batch_chunk_size):
            chunk_end = min(chunk_start + batch_chunk_size, b_new)
            q_chunk = q[i, chunk_start:chunk_end]  # [chunk_size, H/K*W/K, C_head]
            k_chunk = k[i, chunk_start:chunk_end]  # [chunk_size, H/K*W/K, C_head]
            v_chunk = v[i, chunk_start:chunk_end]  # [chunk_size, H/K*W/K, C_head]
            
            scores = torch.matmul(q_chunk, k_chunk.permute(0, 2, 1)) / scale_factor  # [chunk_size, H/K*W/K, H/K*W/K]

            if with_shift:
                scores += attn_mask[chunk_start:chunk_end]

            attn = torch.softmax(scores, dim=-1)

            out_head_chunk = torch.matmul(attn, v_chunk)  # [chunk_size, H/K*W/K, C_head]
            del scores, attn, q_chunk, k_chunk, v_chunk
            gc.collect()
            torch.cuda.empty_cache()
            out_head_chunks.append(out_head_chunk)
        
        out_head = torch.cat(out_head_chunks, dim=0)  # [B*K*K, H/K*W/K, C_head]
        del out_head_chunks
        gc.collect()
        torch.cuda.empty_cache()
        out.append(out_head)
    out = torch.stack(out, dim=0) # [num_heads, B*K*K, H/K*W/K, C_head]
    
    out = out.permute(1, 2, 0, 3) # [B*K*K, H/K*W/K, num_heads, C_head]
    out = out.flatten(2, 3) # [B*K*K, H/K*W/K, C_head*num_heads]

    out = merge_splits(out.view(b_new, h // num_splits, w // num_splits, c),
                       num_splits=num_splits, channel_last=True)  # [B, H, W, C]

    # shift back
    if with_shift:
        out = torch.roll(out, shifts=(shift_size_h, shift_size_w), dims=(1, 2))

    out = out.view(b, -1, c)
    out = out.reshape(b, h*w, num_heads, c_head).permute(0, 2, 1, 3) # [B, num_heads, h*w, c_head]
    out = apply_fn_o(out)  # [B, num_heads, h*w, c_head]
    out = out.permute(0, 2, 1, 3).flatten(2, 3) # [B, H/K*W/K, C]
    
    return out


def compute_urope(intrinsics, pose, feature_height, feature_width, head_dim, device, num_heads, min_depth, max_depth):
    # pose: camera <-- world
    
    pose = pose @ torch.inverse(pose[:, :1])  # camera 1 <- camera 0
    num_cameras = pose.shape[1]
    assert num_cameras in [1, 2]
    batch = pose.shape[0]
    
    viewmats = pose # camera 1 (camera) <-- camera 0 (world)
    Ks = intrinsics[:, None].repeat(1, num_cameras, 1, 1)
    Ks_inv = _invert_K(Ks) # (batch, cameras, 3, 3)
    viewmats_cam2world = _invert_SE3(viewmats) # (batch, cameras, 4, 4) # camera 0 (world) <-- camera 1 (camera)
    
    grid_x, grid_y = torch.meshgrid(torch.arange(feature_width), torch.arange(feature_height), indexing="xy")
    img_x = grid_x + 0.5
    img_y = grid_y + 0.5
    img_z = torch.ones_like(img_x)
    img_xyz = torch.stack([img_x, img_y, img_z], dim=-1).to(device)  # (patch_y, patch_x, 3)
    img_xyz = img_xyz.reshape(-1, 3)[None, None].repeat(batch, num_cameras, 1, 1)  # (batch, cameras, patch_num, 3)
    cam_xyz = img_xyz @ Ks_inv.transpose(-1, -2) # (batch, cameras, patch_num, 3)
    cam_xyz_pad = torch.cat([cam_xyz, torch.ones_like(cam_xyz[..., :1])], dim=-1) # (batch, cameras, patch_num, 4)
    
    raymap = cam_xyz_pad @ viewmats_cam2world.transpose(-1, -2) # (batch, cameras, patch_num, 4)
    raydir = raymap[..., :3] - viewmats_cam2world[..., None, :3, 3]
    
    cam_centers = viewmats_cam2world[..., None, :3, 3].repeat(1, 1, feature_width * feature_height, 1) # (batch, cameras, patch_num, 3)
    raydir = raydir / torch.norm(raydir, dim=-1, keepdim=True)  # (batch, cameras, patch_num, 3)
    
    index = torch.arange(start=0, end=num_heads, step=1, device=device).float()
    bin_size = (max_depth - min_depth) / num_heads
    depths = min_depth + bin_size * index    
    depths = depths[None, :, None]  # [1, depth_num, 1]

    q_img_x = img_x 
    q_img_y = img_y
    q_img_x = q_img_x.flatten(0, 1)[None, None].repeat(batch, num_heads, 1)  # [batch, num_heads, patch_num]
    q_img_y = q_img_y.flatten(0, 1)[None, None].repeat(batch, num_heads, 1)  # [batch, num_heads, patch_num]
    q_img_x = q_img_x.type_as(intrinsics)
    q_img_y = q_img_y.type_as(intrinsics)

    q_coeffs_img_x = _rope_precompute_coeffs_batch(q_img_x, 100, 1, head_dim // 2) 
    q_coeffs_img_y = _rope_precompute_coeffs_batch(q_img_y, 100, 1, head_dim // 2)
    
    kv_cam_x = cam_centers[:, -1, :, 0].reshape(batch, feature_width * feature_height) # [batch, patch_num, 3]
    kv_cam_y = cam_centers[:, -1, :, 1].reshape(batch, feature_width * feature_height) # [batch, patch_num, 3]
    kv_cam_z = cam_centers[:, -1, :, 2].reshape(batch, feature_width * feature_height) # [batch, patch_num, 3]

    kv_ray_x = raydir[:, -1, :, 0].reshape(batch, feature_width * feature_height) # [batch, patch_num, 3]
    kv_ray_y = raydir[:, -1, :, 1].reshape(batch, feature_width * feature_height) # [batch, patch_num, 3]
    kv_ray_z = raydir[:, -1, :, 2].reshape(batch, feature_width * feature_height) # [batch, patch_num, 3]

    kv_x = kv_cam_x[:, None] + kv_ray_x[:, None] * depths  # [batch, depth_num, patch_num]
    kv_y = kv_cam_y[:, None] + kv_ray_y[:, None] * depths  # [batch, depth_num, patch_num]
    kv_z = kv_cam_z[:, None] + kv_ray_z[:, None] * depths  # [batch, depth_num, patch_num]
    
    kv_xyz = torch.stack([kv_x, kv_y, kv_z], dim=-1) # [batch, depth_num, patch_num, 3] 
    kv_img_xyz = kv_xyz @ Ks[:, -1:].transpose(-1, -2) # [batch, depth_num, patch_num, 3] 
    
    kv_img_x = kv_img_xyz[..., 0] / (torch.abs(kv_img_xyz[..., 2]) + 1e-5) # [batch, depth_num, patch_num]
    kv_img_y = kv_img_xyz[..., 1] / (torch.abs(kv_img_xyz[..., 2]) + 1e-5) # [batch, depth_num, patch_num]
    
    kv_img_x = torch.clamp(kv_img_x, min=-64, max=96)
    kv_img_y = torch.clamp(kv_img_y, min=-64, max=96)
    
    kv_coeffs_img_x = _rope_precompute_coeffs_batch(kv_img_x, 100, 1, head_dim // 2)
    kv_coeffs_img_y = _rope_precompute_coeffs_batch(kv_img_y, 100, 1, head_dim // 2)


    transforms_q = [
        (partial(_rope_apply_coeffs, coeffs=q_coeffs_img_x), head_dim // 2),
        (partial(_rope_apply_coeffs, coeffs=q_coeffs_img_y), head_dim // 2),
    ]
    transforms_kv = [
        (partial(_rope_apply_coeffs, coeffs=kv_coeffs_img_x), head_dim // 2),
        (partial(_rope_apply_coeffs, coeffs=kv_coeffs_img_y), head_dim // 2),
    ]
    transforms_o = [
        (partial(_rope_apply_coeffs, coeffs=q_coeffs_img_x, inverse=True), head_dim // 2),
        (partial(_rope_apply_coeffs, coeffs=q_coeffs_img_y, inverse=True), head_dim // 2),
    ]

    apply_fn_q = partial(_apply_block_diagonal, func_size_pairs=transforms_q)
    apply_fn_kv = partial(_apply_block_diagonal, func_size_pairs=transforms_kv)
    apply_fn_o = partial(_apply_block_diagonal, func_size_pairs=transforms_o)

    return apply_fn_q, apply_fn_kv, apply_fn_o
  

def _invert_SE3(transforms: torch.Tensor) -> torch.Tensor:
    """Invert a 4x4 SE(3) matrix."""
    assert transforms.shape[-2:] == (4, 4)
    Rinv = transforms[..., :3, :3].transpose(-1, -2)
    out = torch.zeros_like(transforms)
    out[..., :3, :3] = Rinv
    out[..., :3, 3] = -torch.einsum("...ij,...j->...i", Rinv, transforms[..., :3, 3])
    out[..., 3, 3] = 1.0
    return out


def _lift_K(Ks: torch.Tensor) -> torch.Tensor:
    """Lift 3x3 matrices to homogeneous 4x4 matrices."""
    assert Ks.shape[-2:] == (3, 3)
    out = torch.zeros(Ks.shape[:-2] + (4, 4), device=Ks.device)
    out[..., :3, :3] = Ks
    out[..., 3, 3] = 1.0
    return out


def _invert_K(Ks: torch.Tensor) -> torch.Tensor:
    """Invert 3x3 intrinsics matrices. Assumes no skew."""
    assert Ks.shape[-2:] == (3, 3)
    out = torch.zeros_like(Ks)
    out[..., 0, 0] = 1.0 / Ks[..., 0, 0]
    out[..., 1, 1] = 1.0 / Ks[..., 1, 1]
    out[..., 0, 2] = -Ks[..., 0, 2] / Ks[..., 0, 0]
    out[..., 1, 2] = -Ks[..., 1, 2] / Ks[..., 1, 1]
    out[..., 2, 2] = 1.0
    return out



def _rope_apply_coeffs(
    feats,  # (batch, num_heads, seqlen, feat_dim)
    coeffs, inverse = False,
) -> torch.Tensor:
    """Apply RoPE coefficients to features. We adopt a 'split' ordering
    convention. (in contrast to 'interleaved')"""
    cos, sin = coeffs
    # We allow (cos, sin) to be either with shape (1, 1, seqlen, feat_dim // 2),
    # or (1, 1, seqlen_per_image, feat_dim // 2) and we repeat it to
    # match the shape of feats.
    if cos.shape[2] != feats.shape[2]:
        n_repeats = feats.shape[2] // cos.shape[2]
        cos = cos.repeat(1, 1, n_repeats, 1)
        sin = sin.repeat(1, 1, n_repeats, 1)
    assert len(feats.shape) == len(cos.shape) == len(sin.shape) == 4
    assert cos.shape[-1] == sin.shape[-1] == feats.shape[-1] // 2
    x_in = feats[..., : feats.shape[-1] // 2]
    y_in = feats[..., feats.shape[-1] // 2 :]
    return torch.cat(
        (
            [cos * x_in + sin * y_in, -sin * x_in + cos * y_in]
            if not inverse
            else [cos * x_in - sin * y_in, sin * x_in + cos * y_in]
        ),
        dim=-1,
    )

def _apply_block_diagonal(
    feats: torch.Tensor,  # (..., dim)
    func_size_pairs,
) -> torch.Tensor:
    """Apply a block-diagonal function to an input array.

    Each function is specified as a tuple with form:

        ((Tensor) -> Tensor, int)

    Where the integer is the size of the input to the function.
    """
    funcs, block_sizes = zip(*func_size_pairs)
    assert feats.shape[-1] == sum(block_sizes)
    x_blocks = torch.split(feats, block_sizes, dim=-1)
    out = torch.cat(
        [f(x_block) for f, x_block in zip(funcs, x_blocks)],
        dim=-1,
    )
    assert out.shape == feats.shape, "Input/output shapes should match."
    return out

def _rope_precompute_coeffs_batch(
    positions: torch.Tensor,  # (batch, num_heads, seqlen)
    freq_base: float,
    freq_scale: float,
    feat_dim: int,
):
    """Precompute RoPE coefficients."""
    assert len(positions.shape) == 3
    assert feat_dim % 2 == 0
    num_freqs = feat_dim // 2
    freqs = freq_scale * (
        freq_base
        ** (
            -torch.arange(num_freqs, device=positions.device)[None, None, None, :]
            / num_freqs
        )
    )
    angles = positions[:, :, :, None] * freqs
    # Shape should be: `(batch, num_heads, seqlen, num_freqs)`; we're
    # broadcasting across `batch` and `num_heads`.
    assert angles.shape == (positions.shape[0], positions.shape[1], positions.shape[2], num_freqs)
    return torch.cos(angles), torch.sin(angles)