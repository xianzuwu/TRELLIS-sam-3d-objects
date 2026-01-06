from typing import *
import torch
import torch.nn.functional as F
from easydict import EasyDict as edict
from .sparse_flow_matching import SparseFlowMatchingTrainer

# 确保引用 MOT 版本的 Wrapper 类
# try:
#     from sam3d_objects.model.backbone.tdfy_dit.models.mot_sparse_structure_flow import SparseStructureFlowTdfyWrapper
# except ImportError:
#     pass
from sam3d_objects.model.backbone.tdfy_dit.models.mot_sparse_structure_flow import SparseStructureFlowTdfyWrapper
class SAM3DFlowMatchingTrainer(SparseFlowMatchingTrainer):
    """
    Trainer for SAM 3D (MOT Architecture).
    Handles dictionary-based latents (Shape + Layout), dynamic conditioning, and DDP compatibility.
    """
    
    def training_losses(
        self,
        batch: Dict[str, torch.Tensor],
        cond=None,
        **kwargs
    ) -> Tuple[Dict, Dict]:
        
        # 获取模型实例 (可能是 Adapter 或 DDP Wrapper)
        model = self.models['denoiser']
        
        # ==========================================
        # 1. 准备 GT (x_0) 字典 - (DDP 兼容性增强)
        # ==========================================
        x0_dict = {}
        
        # 尝试获取 input_latent_mappings
        # 优先级: 直接属性 -> DDP (.module) -> Adapter (.model)
        mappings = getattr(model, 'input_latent_mappings', None)
        
        if mappings is None:
            if hasattr(model, 'module'): # PyTorch DDP
                 mappings = getattr(model.module, 'input_latent_mappings', None)
            elif hasattr(model, 'model'): # Custom Adapter
                 mappings = getattr(model.model, 'input_latent_mappings', None)
        
        if mappings is None:
            # 如果依然找不到，说明 Adapter 没有正确暴露属性
            raise AttributeError("Model does not have 'input_latent_mappings'. Check Adapter or DDP wrapping.")

        for key in mappings:
            if key == 'shape':
                # Trellis Dataset 习惯叫 'x_0'，也可能叫 'shape'
                if 'x_0' in batch:
                    x0_dict['shape'] = batch['x_0']
                elif 'shape' in batch:
                    x0_dict['shape'] = batch['shape']
                else:
                    raise ValueError("Batch missing 'x_0' or 'shape' for Shape Token.")
            else:
                # 其他 Layout Tokens
                if key in batch:
                    x0_dict[key] = batch[key]
                else:
                    raise ValueError(f"SAM-3D Model requires latent '{key}', but it is missing in the batch.")

        # ==========================================
        # 2. 采样时间 t
        # ==========================================
        ref_tensor = x0_dict['shape']
        B = ref_tensor.shape[0]
        device = ref_tensor.device
        
        t = self.sample_t(B).to(device).float()
        
        # ==========================================
        # 3. 构造 Flow Matching 目标 (x_t, target)
        # ==========================================
        xt_dict = {}
        target_dict = {}
        
        # 读取配置中的 sigma_min，默认为 1e-5
        sigma_min = getattr(self, 'sigma_min', 1e-5)
        
        for k, v in x0_dict.items():
            noise = torch.randn_like(v)
            t_expand = t.view(B, *([1] * (v.ndim - 1)))
            
            # Flow Matching: x_t = (1 - (1-sigma)*t) * x_0 + t * x_1
            term1 = (1 - (1 - sigma_min) * t_expand) * v
            term2 = t_expand * noise
            xt_dict[k] = term1 + term2
            
            # Vector Field Target: u_t = x_1 - (1-sigma) * x_0
            target_dict[k] = noise - (1 - sigma_min) * v

        # ==========================================
        # 4. 构造条件参数 (Filtering None)
        # ==========================================
        model_kwargs = {}
        # 显式映射 Dataset Key -> Model Arg Key
        cond_map = {
            'image': 'image',
            'mask': 'mask',
            'rgb_image': 'rgb_image',
            'rgb_image_mask': 'rgb_image_mask'
        }
        
        for batch_key, arg_key in cond_map.items():
            val = batch.get(batch_key, None)
            if val is not None:
                model_kwargs[arg_key] = val
        
        # ==========================================
        # 5. 模型前向
        # ==========================================
        # Adapter 需要支持 **kwargs 来接收这些条件
        model_output = model(
            latents_dict=xt_dict,
            t=t * 1000, # DiT Time Scaling
            **model_kwargs
        )
        
        # ==========================================
        # 6. 计算 Loss
        # ==========================================
        terms = edict()
        total_loss = 0.0
        
        # 简单的权重配置 (未来可移入 Config)
        weights = {
            "shape": 1.0,
            "6drotation_normalized": 1.0, 
            "translation": 1.0, 
            "scale": 1.0,
            "translation_scale": 1.0
        }
        
        # 获取当前步数用于控制 Warning 打印频率
        current_step = getattr(self, 'global_step', 0)

        for k in x0_dict.keys():
            if k in model_output:
                loss_k = F.mse_loss(model_output[k], target_dict[k])
                terms[f"mse_{k}"] = loss_k
                total_loss += loss_k * weights.get(k, 1.0)
            else:
                # [优化] 增加警告，防止静默失败
                # 仅在前 5 步打印，避免刷屏
                if current_step < 5: 
                    print(f"[Warning] Key '{k}' found in GT but missing in Model Output. Loss ignored.")

        terms["loss"] = total_loss
        
        return terms, {}