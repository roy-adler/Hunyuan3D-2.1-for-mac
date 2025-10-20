# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import custom_rasterizer_kernel
import torch


def rasterize(pos, tri, resolution, clamp_depth=torch.zeros(0), use_depth_prior=0):
    assert pos.device == tri.device
    
    # Save original device and move tensors to CPU for rasterization
    # (CPU-only rasterizer for macOS compatibility)
    original_device = pos.device
    needs_transfer = original_device.type != 'cpu'
    
    if needs_transfer:
        pos = pos.cpu()
        tri = tri.cpu()
        if clamp_depth.numel() > 0:
            clamp_depth = clamp_depth.cpu()
    
    # Ensure correct data types (float32 for pos, int32 for tri)
    if pos.dtype != torch.float32:
        pos = pos.float()
    if tri.dtype != torch.int32:
        tri = tri.int()
    
    # Rasterize on CPU
    findices, barycentric = custom_rasterizer_kernel.rasterize_image(
        pos[0], tri, clamp_depth, resolution[1], resolution[0], 1e-6, use_depth_prior
    )
    
    # Move results back to original device if needed
    if needs_transfer:
        findices = findices.to(original_device)
        barycentric = barycentric.to(original_device)
    
    return findices, barycentric


def interpolate(col, findices, barycentric, tri):
    f = findices - 1 + (findices == 0)
    vcol = col[0, tri.long()[f.long()]]
    result = barycentric.view(*barycentric.shape, 1) * vcol
    result = torch.sum(result, axis=-2)
    return result.view(1, *result.shape)
