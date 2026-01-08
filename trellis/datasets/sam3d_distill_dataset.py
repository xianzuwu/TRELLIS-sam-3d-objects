import torch
import os
import numpy as np
from torch.utils.data import Dataset, default_collate
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm

class SAM3DDistillDataset(Dataset):
    def __init__(self, path, token_dir, image_dir, image_size=518):
        self.token_dir = token_dir
        self.image_dir = image_dir
        self.image_size = image_size
        
        self.token_files = [f for f in os.listdir(token_dir) if f.endswith('.pt')]
        self.token_files.sort()
        
        self.loads = np.ones(len(self.token_files), dtype=np.float32)
        self.value_range = (0.0, 1.0)
        
        print(f"[Dataset] Found {len(self.token_files)} GT token files in {token_dir}.")
        
        # 计算统计量
        self.stats = self._compute_stats()
        
    def _compute_stats(self):
        print(f"\n[Dataset] ⏳ STATS: Scanning {len(self.token_files)} files to compute Mean/Std...")
        
        shape_accum = []
        trans_scale_log_accum = []
        
        # 扫描所有文件（因为你只有28个，这很快）
        for f in self.token_files:
            gt_path = os.path.join(self.token_dir, f)
            try:
                gt_data = torch.load(gt_path, map_location='cpu', weights_only=False)
                
                # 1. Shape Latent
                s = gt_data['shape'].float()
                # 调试打印：打印第一个文件的形状，确认是否需要 squeeze
                if len(shape_accum) == 0:
                    print(f"[Dataset] DEBUG: First file '{f}' shape tensor size: {s.shape}")
                
                # 展平所有维度，只保留数值分布
                shape_accum.append(s.flatten())
                
                # 2. Translation Scale (Log)
                ts = gt_data['translation_scale'].float().flatten()
                ts = torch.clamp(ts, min=1e-4) # 防止 log(0)
                trans_scale_log_accum.append(torch.log(ts))
                
            except Exception as e:
                # 打印详细错误！
                print(f"[Dataset] ❌ ERROR loading {f}: {e}")
                continue
        
        if len(shape_accum) == 0:
            raise RuntimeError("[Dataset] ❌ CRITICAL: No valid data found for stats! Please check your data paths.")

        # 合并计算
        all_shapes = torch.cat(shape_accum)
        all_ts_log = torch.cat(trans_scale_log_accum)
        
        stats = {
            'shape_mean': all_shapes.mean().item(),
            'shape_std': all_shapes.std().item(),
            'ts_mean': all_ts_log.mean().item(),
            'ts_std': all_ts_log.std().item()
        }
        
        # 强制检查：如果 Std 还是 1.0，说明没算对（除非数据本身就是N(0,1)，但这不可能）
        if abs(stats['shape_std'] - 1.0) < 0.1:
            print(f"[Dataset] ⚠️ WARNING: Calculated Shape Std is {stats['shape_std']:.4f}. Is your data ALREADY normalized?")
        
        print(f"[Dataset] ✅ STATS COMPUTED SUCCESS:")
        print(f"  > Shape:    Mean={stats['shape_mean']:.4f}, Std={stats['shape_std']:.4f}")
        print(f"  > T.Scale:  Mean={stats['ts_mean']:.4f},    Std={stats['ts_std']:.4f}")
        print("-" * 50 + "\n")
        
        return stats

    def __len__(self):
        return len(self.token_files)

    @staticmethod
    def collate_fn(batch, **kwargs):
        return default_collate(batch)
    
    def visualize_sample(self, sample):
        if isinstance(sample, dict):
            if 'rgb_image' in sample: return {'input_image': sample['rgb_image']}
            elif 'image' in sample: return {'input_image': sample['image']}
        return {'input_image': torch.zeros(1, 3, self.image_size, self.image_size)}

    def get_crop_bbox(self, mask_np):
        rows = np.any(mask_np, axis=1)
        cols = np.any(mask_np, axis=0)
        if not np.any(rows) or not np.any(cols):
            return 0, 0, mask_np.shape[1], mask_np.shape[0]
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return cmin, rmin, cmax, rmax

    def preprocess_image_tensor(self, pil_image):
        arr = np.array(pil_image)
        if arr.shape[-1] == 3:
            arr = np.dstack([arr, np.ones_like(arr[..., 0]) * 255])
        x = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        img = x[:3, ...]
        mask = x[3:4, ...]

        C, H, W = img.shape
        if H != W:
            diff = abs(H - W)
            p1 = diff // 2
            p2 = diff - p1
            padding = (p1, p2, 0, 0) if H > W else (0, 0, p1, p2)
            img = F.pad(img, padding, value=1)
            mask = F.pad(mask, padding, value=0)

        img = F.interpolate(img.unsqueeze(0), size=(self.image_size, self.image_size), mode='bilinear', align_corners=False).squeeze(0)
        mask = F.interpolate(mask.unsqueeze(0), size=(self.image_size, self.image_size), mode='nearest').squeeze(0)
        img = img * mask + (1 - mask) 
        return img, mask

    def __getitem__(self, idx):
        token_filename = self.token_files[idx]
        gt_path = os.path.join(self.token_dir, token_filename)
        gt_data = torch.load(gt_path, map_location='cpu', weights_only=False)
        
        # 使用计算出的 Stats 进行归一化
        shape_raw = gt_data['shape'].squeeze(0).float()
        shape_norm = (shape_raw - self.stats['shape_mean']) / (self.stats['shape_std'] + 1e-6)
        
        rot_token = gt_data['6drotation_normalized'].squeeze(0).float()
        trans_token = gt_data['translation'].squeeze(0).float()
        scale_token = gt_data['scale'].squeeze(0).float()
        
        # Translation Scale: Log + Norm + Clamp
        t_scale_raw = gt_data['translation_scale'].squeeze(0).float()
        t_scale_log = torch.log(torch.clamp(t_scale_raw, min=1e-4))
        t_scale_norm = (t_scale_log - self.stats['ts_mean']) / (self.stats['ts_std'] + 1e-6)
        t_scale_norm = torch.clamp(t_scale_norm, min=-5.0, max=5.0)

        file_id = os.path.splitext(token_filename)[0]
        img_name = f"{file_id}.png"
        img_path = os.path.join(self.image_dir, img_name)
        if not os.path.exists(img_path):
             jpg_path = img_path.replace(".png", ".jpg")
             if os.path.exists(jpg_path): img_path = jpg_path
             else: raise FileNotFoundError(f"Image not found: {img_path}")

        pil_image = Image.open(img_path).convert("RGBA")
        global_img_tensor, global_mask_tensor = self.preprocess_image_tensor(pil_image)

        mask_np = np.array(pil_image)[:, :, 3] > 128
        cmin, rmin, cmax, rmax = self.get_crop_bbox(mask_np)
        width, height = cmax - cmin, rmax - rmin
        pad = int(max(width, height) * 0.1)
        cmin, rmin = max(0, cmin - pad), max(0, rmin - pad)
        cmax, rmax = min(pil_image.width, cmax + pad), min(pil_image.height, rmax + pad)

        crop_pil = pil_image.crop((cmin, rmin, cmax, rmax))
        local_img_tensor, local_mask_tensor = self.preprocess_image_tensor(crop_pil)

        return {
            'x_0': shape_norm, 
            '6drotation_normalized': rot_token,
            'translation': trans_token,
            'scale': scale_token,             
            'translation_scale': t_scale_norm, 
            'image': local_img_tensor,
            'mask': local_mask_tensor,
            'rgb_image': global_img_tensor,
            'rgb_image_mask': global_mask_tensor
        }