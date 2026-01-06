import torch
import os
import numpy as np
from torch.utils.data import Dataset
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
        
        # [关键修复] 添加 loads 属性供 BalancedResumableSampler 使用
        # 因为所有样本都是 4096 个点，负载是均衡的，我们全部设为 1
        self.loads = [1] * len(self.token_files)
        
        print(f"[Dataset] Found {len(self.token_files)} GT token files in {token_dir}.")

    def __len__(self):
        return len(self.token_files)

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
        
        return cmin, rmin, cmax, rmax

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
            arr = np.dstack([arr, np.ones_like(arr[..., 0]) * 255])
            
        x = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
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
        shape_token = gt_data['shape'].squeeze(0)
        rot_token = gt_data['6drotation_normalized'].squeeze(0)
        trans_token = gt_data['translation'].squeeze(0)
        scale_token = gt_data['scale'].squeeze(0)
        t_scale_token = gt_data['translation_scale'].squeeze(0)

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
        mask_np = np.array(pil_image)[:, :, 3] > 128 # 阈值 128
        cmin, rmin, cmax, rmax = self.get_crop_bbox(mask_np)
        
        # 添加 10% Padding
        width, height = cmax - cmin, rmax - rmin
        pad = int(max(width, height) * 0.1)
        
        # 确保不越界
        cmin = max(0, cmin - pad)
        rmin = max(0, rmin - pad)
        cmax = min(pil_image.width, cmax + pad)
        rmax = min(pil_image.height, rmax + pad)

        crop_pil = pil_image.crop((cmin, rmin, cmax, rmax))
        local_img_tensor, local_mask_tensor = self.preprocess_image_tensor(crop_pil)

        return {
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