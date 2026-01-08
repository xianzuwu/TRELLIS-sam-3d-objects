import torch
import os
import numpy as np
from torch.utils.data import Dataset, default_collate
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm

class SAM3DDistillDataset(Dataset):
    """
    Dataset for SAM 3D Distillation Training.
    Features:
    - Object-Centric Cropping & Padding.
    - White Background.
    - [NEW] Automatic Latent Normalization (Shape & Translation Scale) to N(0,1).
    """
    def __init__(self, path, token_dir, image_dir, image_size=518):
        self.token_dir = token_dir
        self.image_dir = image_dir
        self.image_size = image_size
        
        self.token_files = [f for f in os.listdir(token_dir) if f.endswith('.pt')]
        self.token_files.sort()
        
        self.loads = np.ones(len(self.token_files), dtype=np.float32)
        self.value_range = (0.0, 1.0)
        
        print(f"[Dataset] Found {len(self.token_files)} GT token files in {token_dir}.")
        
        # [CRITICAL] Compute Statistics for Normalization
        # This ensures inputs to Flow Matching are ~ N(0, 1)
        self.stats = self._compute_stats()
        
    def _compute_stats(self):
        """
        Computes Mean and Std for shape latents and translation_scale.
        """
        print("[Dataset] Computing latent statistics for normalization...")
        
        shape_accum = []
        trans_scale_accum = []
        
        # Limit the number of files to scan if dataset is huge to save time
        # For < 1000 files, full scan is fast.
        scan_files = self.token_files[:1000] 
        
        for f in tqdm(scan_files, desc="Scanning latents"):
            gt_path = os.path.join(self.token_dir, f)
            try:
                # Use weights_only=False to allow loading complex dicts if needed, 
                # though usually False is safer if trusted. 
                # Keeping compatibility with previous loader.
                gt_data = torch.load(gt_path, map_location='cpu', weights_only=False)
                
                # Shape
                shape_accum.append(gt_data['shape'].squeeze(0).float())
                
                # Translation Scale (Log space)
                ts = gt_data['translation_scale'].squeeze(0).float()
                trans_scale_accum.append(torch.log(ts + 1e-6))
            except Exception as e:
                print(f"Error loading {f}: {e}")
                continue
                
        if len(shape_accum) == 0:
            print("[Warning] No data found for stats. Using defaults.")
            return {
                'shape_mean': 0.0, 'shape_std': 1.0,
                'ts_mean': 0.0, 'ts_std': 1.0
            }

        # Stack and calculate
        all_shapes = torch.stack(shape_accum)
        all_ts = torch.stack(trans_scale_accum)
        
        stats = {
            'shape_mean': all_shapes.mean().item(),
            'shape_std': all_shapes.std().item(),
            'ts_mean': all_ts.mean().item(),
            'ts_std': all_ts.std().item()
        }
        
        print(f"[Dataset] Stats computed:")
        print(f"  Shape: Mean={stats['shape_mean']:.4f}, Std={stats['shape_std']:.4f}")
        print(f"  T.Scale (Log): Mean={stats['ts_mean']:.4f}, Std={stats['ts_std']:.4f}")
        
        return stats

    def __len__(self):
        return len(self.token_files)

    @staticmethod
    def collate_fn(batch, **kwargs):
        return default_collate(batch)
    
    def visualize_sample(self, sample):
        if isinstance(sample, dict):
            if 'rgb_image' in sample:
                img = sample['rgb_image']
            elif 'image' in sample:
                img = sample['image']
            else:
                bs = sample.get('x_0', torch.zeros(1)).shape[0]
                img = torch.zeros(bs, 3, self.image_size, self.image_size)
        else:
            img = sample
            
        if not isinstance(img, torch.Tensor):
             return torch.zeros(1, 3, self.image_size, self.image_size)
        
        if img.ndim == 3:
            img = img.unsqueeze(0)
            
        return {'input_image': img}

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
        
        # 1. Shape Normalization (New)
        shape_raw = gt_data['shape'].squeeze(0).float()
        shape_norm = (shape_raw - self.stats['shape_mean']) / (self.stats['shape_std'] + 1e-6)
        
        rot_token = gt_data['6drotation_normalized'].squeeze(0).float()
        trans_token = gt_data['translation'].squeeze(0).float()
        scale_token = gt_data['scale'].squeeze(0).float()
        
        # 2. Translation Scale Log-Normalization
        t_scale_raw = gt_data['translation_scale'].squeeze(0).float()
        t_scale_log = torch.log(t_scale_raw + 1e-6)
        t_scale_normalized = (t_scale_log - self.stats['ts_mean']) / (self.stats['ts_std'] + 1e-6)

        file_id = os.path.splitext(token_filename)[0]
        img_name = f"{file_id}.png"
        img_path = os.path.join(self.image_dir, img_name)
        if not os.path.exists(img_path):
             jpg_path = img_path.replace(".png", ".jpg")
             if os.path.exists(jpg_path):
                 img_path = jpg_path
             else:
                 raise FileNotFoundError(f"Image not found: {img_path}")

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
            'x_0': shape_norm,  # Normalized Shape
            '6drotation_normalized': rot_token,
            'translation': trans_token,
            'scale': scale_token,             
            'translation_scale': t_scale_normalized, # Normalized T.Scale
            'image': local_img_tensor,
            'mask': local_mask_tensor,
            'rgb_image': global_img_tensor,
            'rgb_image_mask': global_mask_tensor
        }