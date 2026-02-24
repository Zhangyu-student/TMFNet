# training_utils.py
import os
import csv
import time
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from visualize import visualize_comparison
from torch.utils.data import DataLoader


def setup_tensorboard(config):
    """设置TensorBoard日志记录器"""
    os.makedirs(config['log_dir'], exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(os.path.join(config['log_dir'], f'run_{timestamp}'))
    print(f"TensorBoard日志保存到: {os.path.join(config['log_dir'], f'run_{timestamp}')}")
    return writer


def setup_csv_logging(config):
    """设置CSV日志记录"""
    os.makedirs(os.path.dirname(config['log_file']) or '.', exist_ok=True)
    csv_file = open(config['log_file'], 'w', newline='')
    fieldnames = ['epoch', 'train_loss', 'val_loss',
                  'val_mse', 'val_mae', 'val_psnr', 'val_sam',
                  'l1_loss', 'spec_loss', 'grad_loss', 'ssim_loss']
    writer_csv = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer_csv.writeheader()
    return csv_file, writer_csv


def create_optimizer(model, config):
    """创建优化器"""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get('lr', 1e-4),
        # weight_decay=config.get('weight_decay', 1e-5)
        weight_decay=config.get('weight_decay', 0)
    )
    return optimizer


def create_scheduler(optimizer, config):
    """创建学习率调度器"""
    scheduler = CosineAnnealingLR(optimizer, T_max=config['num_epochs'])
    return scheduler


def save_model(model, epoch, config, results, best_mse):
    """保存模型检查点"""
    ckpt_path = os.path.join(config['save_dir'], f'model_epoch_{epoch}.pth')
    torch.save(model.state_dict(), ckpt_path)

    # 更新最佳模型
    if results['val_mse'] < best_mse:
        best_mse = results['val_mse']
        best_path = os.path.join(config['save_dir'], 'best.pth')
        torch.save(model.state_dict(), best_path)
    return best_mse


def log_results(writer_csv, epoch, train_loss, results, loss_components):
    """记录结果到CSV"""
    log_data = {
        'epoch': epoch,
        'train_loss': train_loss,
        'val_loss': results['val_loss'],
        'val_mse': results['val_mse'],
        'val_mae': results['val_mae'],
        'val_psnr': results['val_psnr'],
        'val_sam': results['val_sam']
    }
    # 添加损失组件
    for k, v in loss_components.items():
        log_data[k] = v
    writer_csv.writerow(log_data)


def visualize_results(model, test_loader, config, epoch, writer):
    model.eval()
    with torch.no_grad():
        # 创建临时加载器实现随机采样
        temp_loader = DataLoader(
            test_loader.dataset,  # 使用相同数据集
            batch_size=config['batch_size'],  # 保持原批次大小
            shuffle=True,  # 关键：启用随机打乱
            num_workers=0,  # 避免多进程问题
            collate_fn=test_loader.collate_fn  # 保持数据整理方式
        )

        try:
            sample_batch = next(iter(temp_loader))
        except StopIteration:
            return  # 空数据集时安全退出

        test_input = sample_batch["cond_image"].to(config['device'])
        test_target = sample_batch["gt_image"].to(config['device'])
        test_output, att_layer1, att_layer2 = model(test_input)

        # 随机选择批次中的一个样本
        random_idx = torch.randint(0, test_input.size(0), (1,)).item()

        input_vis = test_input[random_idx].cpu()
        output_vis = test_output[random_idx].cpu()
        target_vis = test_target[random_idx].cpu()

        # 修正索引操作的参数，增加 None 判断
        vis_path = visualize_comparison(
            cloudy_input=input_vis,
            output=output_vis,
            target=target_vis,
            epoch=epoch,
            save_dir=config['vis_dir'],
            # 如果 att_layer1 是 None，则直接传递 None；否则执行列表推导
            att_layer1=[a[random_idx].cpu() for a in att_layer1] if att_layer1 is not None else None,
            # 对 att_layer2 做同样处理
            att_layer2=[a[random_idx].cpu() for a in att_layer2] if att_layer2 is not None else None
        )

        if vis_path and os.path.exists(vis_path):
            try:
                from PIL import Image
                import numpy as np
                img = Image.open(vis_path)
                img_tensor = torch.tensor(np.array(img)).permute(2, 0, 1)
                writer.add_image(f'Comparison/epoch_{epoch}', img_tensor, epoch, dataformats='CHW')
            except Exception as e:
                print(f"无法将图像添加到TensorBoard: {e}")