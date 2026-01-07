import torch
import os
import numpy as np
from torch.utils.data import Dataset, default_collate
from PIL import Image
import torch.nn.functional as F

class SAM3DDistillDataset(Dataset):
    """
    Dataset for SAM 3D Distillation Training.
    Correctly implements Object-Centric Cropping and Aspect Ratio Preservation.
    """
    def __init__(self, path, token_dir, image_dir, image_size=518):
        # path 参数是 train.py 传入的 data_dir，这里主要为了兼容接口，实际使用 config 中的 token_dir
        self.token_dir = token_dir
        self.image_dir = image_dir
        self.image_size = image_size
        
        # 获取所有 .pt 文件
        self.token_files = [f for f in os.listdir(token_dir) if f.endswith('.pt')]
        self.token_files.sort()
        
        ## 添加 loads 属性供 BalancedResumableSampler 使用
        self.loads = np.ones(len(self.token_files), dtype=np.float32)
        
        # [关键修复] 添加 value_range 属性
        # 对于图像数据，范围是 [0, 1]；对于 Latent，虽然 Log Space 范围较广，
        # 但 TRELLIS 框架通常期望这里定义为 [0, 1] 或 [-1, 1] 以通过初步检查
        self.value_range = (0.0, 1.0)
        
        print(f"[Dataset] Found {len(self.token_files)} GT token files in {token_dir}.")

    def __len__(self):
        return len(self.token_files)
    
    # [关键修复] 添加 collate_fn 属性
    # train.py 内部会调用这个方法来组织 batch
    @staticmethod
    def collate_fn(batch):
        return default_collate(batch)

    def get_crop_bbox(self, mask_np):
        """
        根据 Alpha Mask 计算 Bounding Box。
        """
        rows = np.any(mask_np, axis=1)
        cols = np.any(mask_np, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            # 如果 Mask 为空，返回全图
            return 0, 0, mask_np.shape[1], mask_np.shape[0]
            
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        # 转换为 Python 原生 int，避免 numpy 标量类型导致 collate 错误
        return int(cmin), int(rmin), int(cmax), int(rmax)

    def preprocess_image_tensor(self, pil_image):
        """
        核心预处理逻辑：
        1. Convert to Tensor
        2. Pad to Square (保持长宽比)
        3. Resize to 518x518
        4. Apply Black Background (image * mask)
        """
        # 1. 转为 Numpy -> Tensor [C, H, W]
        arr = np.array(pil_image)
        
        # 确保有 Alpha 通道
        if arr.shape[-1] == 3:
            # 使用 float32 类型避免 uint8 溢出和类型问题
            alpha = np.ones_like(arr[..., 0], dtype=np.float32) * 255.0
            arr = np.dstack([arr.astype(np.float32), alpha])
        else:
            # 如果已经是 RGBA，也转换为 float32
            arr = arr.astype(np.float32)
            
        x = torch.from_numpy(arr).permute(2, 0, 1) / 255.0
        img = x[:3, ...] # [3, H, W]
        mask = x[3:4, ...] # [1, H, W]

        # 2. Pad to Square (居中填充)
        C, H, W = img.shape
        if H != W:
            diff = abs(H - W)
            p1 = diff // 2
            p2 = diff - p1
            # F.pad 参数顺序: (left, right, top, bottom)
            padding = (p1, p2, 0, 0) if H > W else (0, 0, p1, p2)
            
            img = F.pad(img, padding, value=0)
            mask = F.pad(mask, padding, value=0)

        # 3. Resize (Bilinear for Image, Nearest for Mask)
        # interpolate 需要 [B, C, H, W]
        img = F.interpolate(img.unsqueeze(0), size=(self.image_size, self.image_size), mode='bilinear', align_corners=False).squeeze(0)
        mask = F.interpolate(mask.unsqueeze(0), size=(self.image_size, self.image_size), mode='nearest').squeeze(0)

        # 4. 背景处理: 黑底混合 (SAM 3D / TRELLIS 常用)
        img = img * mask 
        
        return img, mask

    def __getitem__(self, idx):
        token_filename = self.token_files[idx]
        file_id = os.path.splitext(token_filename)[0]
        
        # ==========================================
        # 1. Load GT Tokens
        # ==========================================
        gt_path = os.path.join(self.token_dir, token_filename)
        # 使用 map_location='cpu' 避免多进程 dataloader 显存爆炸
        gt_data = torch.load(gt_path, map_location='cpu')
        
        # 移除 Batch 维度
        # 确保所有字段都被正确处理，避免任何 uint8 类型或意外的形状
        shape_token = gt_data['shape'].squeeze(0).float()  # 确保是 float32
        rot_token = gt_data['6drotation_normalized'].squeeze(0).float()
        trans_token = gt_data['translation'].squeeze(0).float()
        scale_token = gt_data['scale'].squeeze(0).float()
        # translation_scale 是 (1, 1, 1)，squeeze(0) 后是 (1, 1)，需要确保正确
        t_scale_token = gt_data['translation_scale'].squeeze(0).float()
        
        # 验证所有 token 都是 float32 类型，避免任何类型问题
        assert shape_token.dtype == torch.float32, f"shape_token dtype is {shape_token.dtype}, expected float32"
        assert rot_token.dtype == torch.float32, f"rot_token dtype is {rot_token.dtype}, expected float32"
        assert trans_token.dtype == torch.float32, f"trans_token dtype is {trans_token.dtype}, expected float32"
        assert scale_token.dtype == torch.float32, f"scale_token dtype is {scale_token.dtype}, expected float32"
        assert t_scale_token.dtype == torch.float32, f"t_scale_token dtype is {t_scale_token.dtype}, expected float32"

        # ==========================================
        # 2. Load Image
        # ==========================================
        img_name = f"{file_id}.png"
        img_path = os.path.join(self.image_dir, img_name)
        
        if not os.path.exists(img_path):
             jpg_path = img_path.replace(".png", ".jpg")
             if os.path.exists(jpg_path):
                 img_path = jpg_path
             else:
                 raise FileNotFoundError(f"Image not found: {img_path}")

        pil_image = Image.open(img_path).convert("RGBA")

        # ==========================================
        # 3. Dual Stream Processing
        # ==========================================
        
        # [A] Global View (rgb_image)
        # 全图 -> Pad to Square -> Resize
        global_img_tensor, global_mask_tensor = self.preprocess_image_tensor(pil_image)

        # [B] Local View (image)
        # Crop -> Pad to Square -> Resize
        
        # 计算 Mask BBox
        # 使用 float32 类型避免 uint8 类型问题
        arr_for_mask = np.array(pil_image).astype(np.float32)
        mask_np = arr_for_mask[:, :, 3] > 128.0 # 阈值 128，使用 float 比较
        cmin, rmin, cmax, rmax = self.get_crop_bbox(mask_np)
        
        # 添加 10% Padding
        width, height = cmax - cmin, rmax - rmin
        pad = int(max(width, height) * 0.1)
        
        # 确保不越界（确保所有值都是 Python int）
        cmin = int(max(0, cmin - pad))
        rmin = int(max(0, rmin - pad))
        cmax = int(min(pil_image.width, cmax + pad))
        rmax = int(min(pil_image.height, rmax + pad))

        crop_pil = pil_image.crop((cmin, rmin, cmax, rmax))
        local_img_tensor, local_mask_tensor = self.preprocess_image_tensor(crop_pil)

        result = {
            # GT
            'x_0': shape_token, 
            '6drotation_normalized': rot_token,
            'translation': trans_token,
            'scale': scale_token,             
            'translation_scale': t_scale_token,
            
            # Condition (Dual Stream)
            'image': local_img_tensor,           # [3, 518, 518]
            'mask': local_mask_tensor,           # [1, 518, 518]
            'rgb_image': global_img_tensor,      # [3, 518, 518]
            'rgb_image_mask': global_mask_tensor # [1, 518, 518]
        }
        
        # 确保所有值都是 torch.Tensor 类型，避免任何 numpy 数组或意外类型
        for k, v in result.items():
            if not isinstance(v, torch.Tensor):
                raise TypeError(f"Field '{k}' is not a torch.Tensor: {type(v)}")
            if v.dtype == torch.uint8:
                raise TypeError(f"Field '{k}' has uint8 dtype, which may cause issues. Shape: {v.shape}")
        
        return result
    
    def visualize_sample(self, sample):
        """
        Visualize a sample for snapshot_dataset.
        Returns the image tensor for visualization.
        Note: sample is already a batch (from dataloader), so we return the batch tensor directly.
        """
        # sample 已经是 batch 数据，直接返回图像 tensor
        # 确保返回的是 torch.Tensor，形状为 [B, C, H, W]
        if isinstance(sample, dict):
            if 'rgb_image' in sample:
                img = sample['rgb_image']
            elif 'image' in sample:
                img = sample['image']
            else:
                # 如果没有图像，返回一个占位符 batch
                batch_size = sample.get('x_0', torch.zeros(1, 4096, 8)).shape[0]
                img = torch.zeros(batch_size, 3, 518, 518)
        else:
            # 如果不是 dict，假设 sample 本身就是图像 tensor
            img = sample
        
        # 确保是 torch.Tensor 类型
        if not isinstance(img, torch.Tensor):
            raise TypeError(f"visualize_sample returned {type(img)}, expected torch.Tensor")
        
        # 确保形状正确 [B, C, H, W] 或 [C, H, W]
        if img.ndim == 3:
            # 如果是 [C, H, W]，添加 batch 维度
            img = img.unsqueeze(0)
        elif img.ndim != 4:
            raise ValueError(f"visualize_sample returned tensor with shape {img.shape}, expected [B, C, H, W] or [C, H, W]")
        
        return img