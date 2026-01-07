from typing import *
import os
import copy
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from easydict import EasyDict as edict
from .sparse_flow_matching import SparseFlowMatchingTrainer

try:
    from sam3d_objects.model.backbone.tdfy_dit.models.mot_sparse_structure_flow import SparseStructureFlowTdfyWrapper
except ImportError:
    pass

class SAM3DFlowMatchingTrainer(SparseFlowMatchingTrainer):
    """
    Trainer for SAM 3D (MOT Architecture).
    """
    
    # [CRITICAL FIX] 使用 **batch 接收所有关键字参数，将其打包为字典
    # 之前定义的 (self, batch, ...) 会导致找不到名为 'batch' 的参数而报错
    def training_losses(
        self,
        **batch
    ) -> Tuple[Dict, Dict]:
        
        model = self.models['denoiser']
        
        # 1. 准备 GT (x_0)
        x0_dict = {}
        mappings = getattr(model, 'input_latent_mappings', None)
        if mappings is None:
            if hasattr(model, 'module'): mappings = getattr(model.module, 'input_latent_mappings', None)
            elif hasattr(model, 'model'): mappings = getattr(model.model, 'input_latent_mappings', None)
        
        if mappings is None:
            raise AttributeError("Model does not have 'input_latent_mappings'.")

        for key in mappings:
            if key == 'shape':
                # Dataset 返回的是 'x_0'，这里做一下兼容
                if 'x_0' in batch: x0_dict['shape'] = batch['x_0']
                elif 'shape' in batch: x0_dict['shape'] = batch['shape']
            else:
                if key in batch: x0_dict[key] = batch[key]

        # 2. 采样时间 t
        # 使用 x0_dict 中任意一个 tensor 来获取 batch size 和 device
        ref_tensor = x0_dict['shape'] 
        B = ref_tensor.shape[0]
        t = self.sample_t(B).to(ref_tensor.device).float()
        
        # 3. Flow Matching (加噪)
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

        # 4. 条件注入
        model_kwargs = {}
        cond_map = {'image': 'image', 'mask': 'mask', 'rgb_image': 'rgb_image', 'rgb_image_mask': 'rgb_image_mask'}
        for bk, ak in cond_map.items():
            if batch.get(bk) is not None:
                model_kwargs[ak] = batch[bk]
        
        # 5. 模型前向
        model_output = model(latents_dict=xt_dict, t=t * 1000, **model_kwargs)
        
        # 6. Loss 计算
        terms = edict()
        total_loss = 0.0
        weights = {"shape": 1.0, "6drotation_normalized": 1.0, "translation": 1.0, "scale": 1.0, "translation_scale": 1.0}
        
        current_step = getattr(self, 'step', 0)

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
    def run_snapshot(
        self,
        num_samples: int = 64,
        batch_size: int = 4,
        verbose: bool = False,
    ) -> Dict:
        """
        Custom implementation for visualizing input conditions.
        """
        collate_fn = self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None
        
        dataloader = DataLoader(
            copy.deepcopy(self.dataset),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn,
        )

        sample_dict = {}
        vis_images = []
        vis_masks = []
        
        count = 0
        for data in dataloader:
            if count >= num_samples:
                break
                
            if 'image' in data:
                vis_images.append(data['image'])
            
            if 'mask' in data:
                mask = data['mask']
                if mask.shape[1] == 1: mask = mask.repeat(1, 3, 1, 1)
                vis_masks.append(mask)

            count += batch_size

        current_step = getattr(self, 'step', 0)

        if len(vis_images) > 0:
            all_imgs = torch.cat(vis_images, dim=0)[:num_samples]
            sample_dict['cond_image'] = {'value': all_imgs, 'type': 'image'}
            
            if self.is_master:
                save_dir = os.path.join(self.output_dir, 'samples', f'step_{current_step:06d}')
                os.makedirs(save_dir, exist_ok=True)
                vutils.save_image(all_imgs, os.path.join(save_dir, 'inputs.jpg'), nrow=4, padding=2)
                print(f"[Snapshot] Saved inputs to {save_dir}")

        if len(vis_masks) > 0:
            all_masks = torch.cat(vis_masks, dim=0)[:num_samples]
            sample_dict['cond_mask'] = {'value': all_masks, 'type': 'image'}
        
        return sample_dict