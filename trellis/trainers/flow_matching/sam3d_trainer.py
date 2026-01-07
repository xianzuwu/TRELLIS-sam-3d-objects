from typing import *
import os
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from easydict import EasyDict as edict
from .sparse_flow_matching import SparseFlowMatchingTrainer

# 尝试导入 MOT Wrapper
# try:
#     from sam3d_objects.model.backbone.tdfy_dit.models.mot_sparse_structure_flow import SparseStructureFlowTdfyWrapper
# except ImportError:
#     pass

from sam3d_objects.model.backbone.tdfy_dit.models.mot_sparse_structure_flow import SparseStructureFlowTdfyWrapper

class SAM3DFlowMatchingTrainer(SparseFlowMatchingTrainer):
    """
    Trainer for SAM 3D (MOT Architecture).
    Handles dictionary-based latents, dynamic conditioning, and safe input visualization.
    """
    
    def training_losses(
        self,
        batch: Dict[str, torch.Tensor],
        cond=None,
        **kwargs
    ) -> Tuple[Dict, Dict]:
        
        model = self.models['denoiser']
        
        # ==========================================
        # 1. 准备 GT (x_0)
        # ==========================================
        x0_dict = {}
        # 兼容 DDP 和 普通模式获取 mapping
        mappings = getattr(model, 'input_latent_mappings', None)
        if mappings is None:
            if hasattr(model, 'module'): 
                mappings = getattr(model.module, 'input_latent_mappings', None)
            elif hasattr(model, 'model'): 
                mappings = getattr(model.model, 'input_latent_mappings', None)
        
        if mappings is None:
            raise AttributeError("Model does not have 'input_latent_mappings'.")

        for key in mappings:
            if key == 'shape':
                if 'x_0' in batch: x0_dict['shape'] = batch['x_0']
                elif 'shape' in batch: x0_dict['shape'] = batch['shape']
            else:
                if key in batch: x0_dict[key] = batch[key]

        # ==========================================
        # 2. 采样时间 t
        # ==========================================
        ref_tensor = x0_dict['shape']
        B = ref_tensor.shape[0]
        t = self.sample_t(B).to(ref_tensor.device).float()
        
        # ==========================================
        # 3. Flow Matching (加噪)
        # ==========================================
        xt_dict = {}
        target_dict = {}
        sigma_min = getattr(self, 'sigma_min', 1e-5)
        
        for k, v in x0_dict.items():
            noise = torch.randn_like(v)
            t_expand = t.view(B, *([1] * (v.ndim - 1)))
            
            # x_t = (1 - (1-sigma)t) * x_0 + t * x_1
            xt_dict[k] = (1 - (1 - sigma_min) * t_expand) * v + t_expand * noise
            # u_t = x_1 - (1-sigma) * x_0
            target_dict[k] = noise - (1 - sigma_min) * v

        # ==========================================
        # 4. 条件注入 (Filtering)
        # ==========================================
        model_kwargs = {}
        cond_map = {
            'image': 'image', 
            'mask': 'mask', 
            'rgb_image': 'rgb_image', 
            'rgb_image_mask': 'rgb_image_mask'
        }
        for batch_key, arg_key in cond_map.items():
            if batch.get(batch_key) is not None:
                model_kwargs[arg_key] = batch[batch_key]
        
        # ==========================================
        # 5. 模型前向
        # ==========================================
        model_output = model(
            latents_dict=xt_dict,
            t=t * 1000, 
            **model_kwargs
        )
        
        # ==========================================
        # 6. Loss 计算
        # ==========================================
        terms = edict()
        total_loss = 0.0
        
        # 权重设置
        weights = {
            "shape": 1.0,
            "6drotation_normalized": 1.0, 
            "translation": 1.0, 
            "scale": 1.0, 
            "translation_scale": 1.0
        }
        
        current_step = getattr(self, 'global_step', 0)

        for k in x0_dict.keys():
            if k in model_output:
                loss_k = F.mse_loss(model_output[k], target_dict[k])
                terms[f"mse_{k}"] = loss_k
                total_loss += loss_k * weights.get(k, 1.0)
            else:
                if current_step < 5: 
                    print(f"[Warning] Key '{k}' missing in output.")

        terms["loss"] = total_loss
        
        return terms, {}

    @torch.no_grad()
    def sample(self, batch, **kwargs):
        """
        Safe Sampling: Visualization Only.
        Since VAE is missing, we only visualize input conditions (Image/Mask)
        to verify data correctness.
        """
        # 1. 准备目录
        sample_dir = os.path.join(self.output_dir, 'samples', f'step_{self.global_step:06d}')
        os.makedirs(sample_dir, exist_ok=True)
        
        # 2. 获取文件名 (处理 Tensor 类型的 name)
        names = batch.get('name', [str(i) for i in range(len(batch['x_0']))])
        if isinstance(names, torch.Tensor):
            # 如果是 Tensor，尝试转 list，通常 DataLoader 对 string list 不会转 tensor
            # 但如果使用了 default_collate 且无法处理 string，可能会有问题
            # 这里做一个容错，如果实在拿不到名字就用索引
            names = [str(i) for i in range(len(names))]
            
        # 3. 可视化输入 (Image, Mask)
        keys_to_vis = ['image', 'rgb_image', 'mask', 'rgb_image_mask']
        
        for k in keys_to_vis:
            if k in batch:
                img_tensor = batch[k].detach().cpu() # [B, C, H, W]
                
                # 如果是 Mask (1通道)，repeat 成 3 通道以便 save_image
                if img_tensor.shape[1] == 1:
                    img_tensor = img_tensor.repeat(1, 3, 1, 1)
                
                # 保存前 8 张
                # save_image 会自动处理 [0, 1] 的 float，不需要额外乘 255
                save_path = os.path.join(sample_dir, f"vis_{k}.png")
                vutils.save_image(img_tensor[:8], save_path, nrow=4, padding=2)
        
        if self.global_step % 100 == 0:
            print(f"[Sampling] Saved input visualizations to {sample_dir}")