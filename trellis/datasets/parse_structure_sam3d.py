import os
import json
import numpy as np
import torch
from PIL import Image
from .sparse_structure_latent import SparseStructureLatent

# --- SAM 3D 硬编码统计量 (来自https://github.com/facebookresearch/sam-3d-objects/sam3d_objects/pipeline/inference_utils.py) ---
# 这些统计量用于将物理属性归一化到模型易于学习的分布
ROTATION_6D_MEAN = torch.tensor([-0.0637, 0.0084, 0.0002, 0.0007, -0.0031, 0.5166])
ROTATION_6D_STD = torch.tensor([0.6657, 0.6787, 0.3035, 0.4395, 0.3982, 0.6176])

class SAM3DSparseStructureDataset(SparseStructureLatent):
    """
    SAM 3D Dataset Integration.
    Features:
    - Dual-stream image input (Full Image + Cropped Image).
    - Normalized Layout Attributes (Log scale, Standardized Rotation).
    - Latent Normalization (via parent class).
    """
    def __init__(self, roots, image_size=518, **kwargs):
        # 注意: 务必在 kwargs 或 config 中传入 'normalization' 参数
        # 以便父类 SparseStructureLatent 正确归一化 VAE Latent
        super().__init__(roots, **kwargs)
        self.image_size = image_size

    def get_instance(self, root, instance):
        # 1. 获取 Shape Latent (x_0)
        # 父类 get_instance 会处理加载和 normalization (如果 init 传了参)
        pack = super().get_instance(root, instance)
        
        # [Fix] 调整维度为 Channels Last [D, H, W, C] 
        # TRELLIS 原始 storage 可能是 [C, D, H, W] 或 [D, H, W, C]，需统一
        if isinstance(pack['x_0'], np.ndarray):
            pack['x_0'] = torch.from_numpy(pack['x_0'])
        pack['x_0'] = pack['x_0'].float()
        
        # 假设原始是 [8, 16, 16, 16] (C, D, H, W)，转为 [16, 16, 16, 8]
        # 很多 Sparse Transformer 喜欢 Channels Last
        if pack['x_0'].shape[0] == 8 and pack['x_0'].ndim == 4: 
             pack['x_0'] = pack['x_0'].permute(1, 2, 3, 0)

        # 2. 获取 Layout GT (物理空间 -> 模型空间)
        # [假设] JSON 中的数据是 Linear Scale 和 物理 Rotation
        layout_path = os.path.join(root, 'layout', f'{instance}.json')
        
        # 默认值 (Identity / Zero)
        layout_data = {
            'rotation_6d': [1, 0, 0, 0, 1, 0], 
            'translation': [0, 0, 0],
            'scale': [1, 1, 1], 
            'translation_scale': [1]
        }
        
        if os.path.exists(layout_path):
            with open(layout_path, 'r') as f:
                layout_data.update(json.load(f))
        
        # [Fix] Layout 归一化处理逻辑
        
        # A. Rotation: 6D -> Normalize
        rot_6d = torch.tensor(layout_data['rotation_6d'], dtype=torch.float32)
        rot_norm = (rot_6d - ROTATION_6D_MEAN) / ROTATION_6D_STD
        pack['6drotation_normalized'] = rot_norm.view(1, 6)

        # B. Scale: Linear -> Log
        # clamp 防止 log(0)
        scale_val = torch.tensor(layout_data['scale'], dtype=torch.float32)
        pack['scale'] = torch.log(scale_val.clamp(min=1e-6)).view(1, 3)

        # C. Translation Scale: Linear -> Log
        t_scale_val = torch.tensor(layout_data['translation_scale'], dtype=torch.float32)
        pack['translation_scale'] = torch.log(t_scale_val.clamp(min=1e-6)).view(1, 1)

        # D. Translation: 保持原值 
        # (假设数据预处理阶段已做 SSI 归一化，如果未做需在此处除以 scene_scale)
        pack['translation'] = torch.tensor(layout_data['translation'], dtype=torch.float32).view(1, 3)

        # 3. 图像处理 (Full + Crop)
        image_root = os.path.join(root, 'renders_cond', instance)
        
        if os.path.exists(os.path.join(image_root, 'transforms.json')):
            with open(os.path.join(image_root, 'transforms.json')) as f:
                meta = json.load(f)
            
            view_idx = np.random.randint(len(meta['frames']))
            frame_meta = meta['frames'][view_idx]
            image_path = os.path.join(image_root, frame_meta['file_path'])
            
            pil_image = Image.open(image_path)
            
            # [A] 准备 Full Image (rgb_image) - 仅 Resize，不 Crop
            # 这提供了 Global Context
            full_img = pil_image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
            full_rgb = torch.from_numpy(np.array(full_img.convert('RGB'))).permute(2, 0, 1).float() / 255.0
            full_mask = torch.from_numpy(np.array(full_img.getchannel(3))).float() / 255.0
            full_mask = full_mask.unsqueeze(0)
            
            # 白底预乘 (Standard for DINO)
            pack['rgb_image'] = full_rgb * full_mask + (1 - full_mask)
            pack['rgb_image_mask'] = full_mask 

            # [B] 准备 Cropped Image (image) - BBox Crop
            # 这提供了 Local Detail
            alpha_np = np.array(pil_image.getchannel(3))
            coords = np.nonzero(alpha_np)
            if coords[0].size > 0:
                y_min, y_max = coords[0].min(), coords[0].max()
                x_min, x_max = coords[1].min(), coords[1].max()
                center = [(x_min + x_max) / 2, (y_min + y_max) / 2]
                size = max(x_max - x_min, y_max - y_min) / 2 * 1.2 
                left = int(center[0] - size)
                upper = int(center[1] - size)
                right = int(center[0] + size)
                lower = int(center[1] + size)
                pil_crop = pil_image.crop((left, upper, right, lower))
            else:
                pil_crop = pil_image
                
            pil_crop = pil_crop.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
            crop_rgb = torch.from_numpy(np.array(pil_crop.convert('RGB'))).permute(2, 0, 1).float() / 255.0
            crop_mask = torch.from_numpy(np.array(pil_crop.getchannel(3))).float() / 255.0
            crop_mask = crop_mask.unsqueeze(0)
            
            pack['image'] = crop_rgb * crop_mask + (1 - crop_mask)
            pack['mask'] = crop_mask 
        else:
            # Fallback
            zeros = torch.zeros(3, self.image_size, self.image_size)
            zeros_mask = torch.zeros(1, self.image_size, self.image_size)
            pack['rgb_image'] = zeros
            pack['rgb_image_mask'] = zeros_mask
            pack['image'] = zeros
            pack['mask'] = zeros_mask

        return pack