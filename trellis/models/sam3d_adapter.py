import torch
import torch.nn as nn
from easydict import EasyDict as edict

from sam3d_objects.model.backbone.tdfy_dit.models.mot_sparse_structure_flow import SparseStructureFlowTdfyWrapper
from sam3d_objects.model.backbone.dit.embedder.embedder_fuser import EmbedderFuser
from sam3d_objects.model.backbone.dit.embedder.dino import Dino 
from sam3d_objects.model.backbone.tdfy_dit.models.mm_latent import (
    Latent, 
    ShapePositionEmbedder, 
    RandomPositionEmbedder
)

class SAM3DStructureFlowAdapter(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        cfg = edict(kwargs)

        # 1. Condition Embedders
        dino_model_name = cfg.get('dino_model', 'dinov2_vitl14') 
        self.img_embedder = Dino(dino_model=dino_model_name)
        self.mask_embedder = Dino(dino_model=dino_model_name)

        # 2. Embedder Fuser (Shared Pos)
        embedder_list = [
            (self.img_embedder, [('image', 'image_pos'), ('rgb_image', 'image_pos')]), 
            (self.mask_embedder, [('mask_image', 'image_pos'), ('rgb_image_mask', 'image_pos')]) 
        ]

        self.condition_embedder = EmbedderFuser(
            embedder_list=embedder_list,
            use_pos_embedding="learned", 
            projection_net_hidden_dim_multiplier=4.0
        )
        self.cond_dim = max(self.img_embedder.embed_dim, self.mask_embedder.embed_dim)

        # 3. Latent Mappings
        shape_dim = cfg.get('in_channels', 8) 
        model_dim = cfg.model_channels 
        
        shape_pos_factory = ShapePositionEmbedder(
            model_channels=model_dim, resolution=cfg.get('resolution', 16), patch_size=1
        )
        layout_pos_factory = RandomPositionEmbedder(model_channels=model_dim, token_len=1)

        self.latent_mapping = nn.ModuleDict({
            "shape": Latent(shape_dim, model_dim, shape_pos_factory),
            "6drotation_normalized": Latent(6, model_dim, layout_pos_factory),
            "translation": Latent(3, model_dim, layout_pos_factory),
            "scale": Latent(3, model_dim, layout_pos_factory),
            "translation_scale": Latent(1, model_dim, layout_pos_factory),
        })

        # 4. MOT Backbone
        latent_share_transformer_cfg = {
            "6drotation_normalized": ["6drotation_normalized", "translation", "scale", "translation_scale"],
            "mixed_tokens": ["shape"] 
        }

        self.model = SparseStructureFlowTdfyWrapper(
            latent_mapping=self.latent_mapping,
            latent_share_transformer=latent_share_transformer_cfg,
            in_channels=shape_dim,
            model_channels=model_dim,
            cond_channels=self.cond_dim, 
            out_channels=shape_dim, 
            num_blocks=cfg.num_blocks,
            num_heads=cfg.num_heads,
            mlp_ratio=cfg.mlp_ratio,
            condition_embedder=self.condition_embedder,
            use_fp16=cfg.get('use_fp16', True),
            use_checkpoint=cfg.get('use_checkpoint', True)
        )

    # [关键] 暴露属性给 Trainer
    @property
    def input_latent_mappings(self):
        return self.model.input_latent_mappings

    def forward(self, latents_dict, t, **kwargs):
        """
        Supports **kwargs for flexible conditioning (mask -> mask_image).
        """
        forward_kwargs = {}
        
        if 'image' in kwargs: forward_kwargs['image'] = kwargs['image']
        
        if 'mask' in kwargs: forward_kwargs['mask_image'] = kwargs['mask']
        elif 'mask_image' in kwargs: forward_kwargs['mask_image'] = kwargs['mask_image']
            
        if 'rgb_image' in kwargs: forward_kwargs['rgb_image'] = kwargs['rgb_image']
        if 'rgb_image_mask' in kwargs: forward_kwargs['rgb_image_mask'] = kwargs['rgb_image_mask']

        output_dict = self.model(
            latents_dict=latents_dict, 
            t=t,
            **forward_kwargs
        )
        return output_dict