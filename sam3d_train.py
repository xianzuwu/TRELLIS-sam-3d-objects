import os
import torch
import argparse
from torch.utils.data import DataLoader

# 引用模块
from trellis.models.sam3d_adapter import SAM3DStructureFlowAdapter
from trellis.trainers.flow_matching.sam3d_trainer import SAM3DFlowMatchingTrainer
from trellis.datasets.sam3d_distill_dataset import SAM3DDistillDataset

def main():
    parser = argparse.ArgumentParser()
    
    # [配置] 这里的默认路径已更新为你提供的绝对路径
    parser.add_argument("--token_dir", type=str, 
                        default="/playpen-shared/xianfeng/Projects/sam-3d-objects/training_gt_tokens", 
                        help="Path to extracted .pt tokens")
    
    parser.add_argument("--image_dir", type=str, 
                        default="/playpen-shared/xianfeng/Projects/sam-3d-objects/notebook/images/shutterstock_stylish_kidsroom_1640806567", 
                        help="Path to source images")
    
    parser.add_argument("--output_dir", type=str, default="outputs/sam3d_distill_experiment", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size") 
    parser.add_argument("--num_steps", type=int, default=1000, help="Total training steps")
    args = parser.parse_args()

    # 1. 配置模型参数 (Stage 1 Config)
    # ---------------------------------------------------------
    model_cfg = {
        "in_channels": 8,           # VAE latent channels (SAM3D 默认是 8)
        "model_channels": 1024,     # ViT-L 维度
        "out_channels": 8,
        "num_blocks": 24,           # DiT-L/2 Depth
        "num_heads": 16,
        "mlp_ratio": 4,
        "patch_size": 1,            # Latent 已经是序列，不需要再 patchify
        "resolution": 16,
        "dino_model": "dinov2_vitl14", # 使用 Large DINO
        "use_fp16": True,
        "use_checkpoint": True      # 开启梯度检查点节省显存
    }

    print(">>> Initializing SAM3D Adapter Model (ViT-L)...")
    # 初始化模型
    model = SAM3DStructureFlowAdapter(**model_cfg)
    model.cuda()
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f">>> Model loaded. Parameters: {param_count / 1e6:.2f} M")

    # 2. 准备数据集
    # ---------------------------------------------------------
    print(f">>> Loading Distill Dataset...")
    print(f"    Tokens: {args.token_dir}")
    print(f"    Images: {args.image_dir}")
    
    # 简单的路径存在性检查
    if not os.path.exists(args.token_dir):
        raise FileNotFoundError(f"Token directory not found: {args.token_dir}")
    if not os.path.exists(args.image_dir):
        raise FileNotFoundError(f"Image directory not found: {args.image_dir}")

    dataset = SAM3DDistillDataset(
        token_dir=args.token_dir,
        image_dir=args.image_dir
    )
    
    # DataLoader 配置
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4,
        drop_last=True, # 丢弃最后不足一个 batch 的数据
        pin_memory=True
    )
    
    print(f">>> Dataset loaded. {len(dataset)} samples found.")

    # 3. 初始化 Trainer
    # ---------------------------------------------------------
    training_cfg = {
        "optimizer": {
            "name": "AdamW",
            "args": {"lr": 1e-4, "weight_decay": 0.05} 
        },
        "scheduler": {
            "name": "cosine_with_restart", 
            "args": {"warmup_steps": 100}
        }
    }
    
    from easydict import EasyDict
    cfg = EasyDict({
        "output_dir": args.output_dir,
        "solver": training_cfg
    })

    models_dict = {"denoiser": model}
    
    trainer = SAM3DFlowMatchingTrainer(
        cfg=cfg,
        models=models_dict,
        dataloaders={"train": dataloader},
        device_mesh=None 
    )

    # 4. 开始训练循环
    # ---------------------------------------------------------
    print(">>> Starting Distillation Training...")
    
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 构造无限迭代器
    def infinite_loader(loader):
        while True:
            for batch in loader:
                yield batch
    
    train_iter = infinite_loader(dataloader)

    for step in range(1, args.num_steps + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = infinite_loader(dataloader)
            batch = next(train_iter)
        except Exception as e:
            print(f"Error loading batch: {e}")
            continue
        
        # Move to GPU
        batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # Loss Computation
        losses, _ = trainer.training_losses(batch)
        loss = losses["loss"]
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        # 日志打印
        if step % 10 == 0:
            print(f"[Step {step:04d}/{args.num_steps}] "
                  f"Loss: {loss.item():.6f} | "
                  f"Shape: {losses.get('mse_shape', 0):.6f} | "
                  f"Rot: {losses.get('mse_6drotation_normalized', 0):.6f} | "
                  f"Scale: {losses.get('mse_scale', 0):.6f}")
        
        # 保存 Checkpoint
        if step % 500 == 0:
            save_path = os.path.join(args.output_dir, f"checkpoint_{step:06d}.pt")
            torch.save(model.state_dict(), save_path)
            print(f">>> Saved checkpoint to {save_path}")

    print(">>> Training finished!")

if __name__ == "__main__":
    main()