import os
import torch
from dataset import Sen2_MTC_New_Multi, MultipleDataset
from models.mamba_test import TemporalMambaFusionNet  # 根据实际路径调整
from torch.utils.data import DataLoader, random_split

def create_dataloaders(config):
    """根据配置创建训练和测试数据加载器"""
    if config['dataset_type'] == 'new_multi':
        dataset_class = Sen2_MTC_New_Multi
        # 创建数据集实例
        train_dataset = dataset_class(data_root=config['data_root'], mode="train")
        test_dataset = dataset_class(data_root=config['data_root'], mode="test")
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config.get('num_workers', 4),
            pin_memory=True,
            persistent_workers=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config.get('num_workers', 2),
            pin_memory=True
        )

    elif config['dataset_type'] == 'old_multi':
        dataset_class = MultipleDataset
        total_data = dataset_class(
            data_root=os.path.join(config['data_root'], "multipleImage"),
            band=3,
        )
        print(len(total_data))
        train_data, val_data, test_data = random_split(
            dataset=total_data,
            lengths=(2504, 313, 313),
            generator=torch.Generator().manual_seed(2022),
        )

        train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True,
                                  num_workers= config.get('num_workers', 4), drop_last=False, pin_memory=True, persistent_workers=True)
        test_loader = DataLoader(test_data, batch_size=config['batch_size'], shuffle=False,
                                num_workers=config.get('num_workers', 2), drop_last=False, pin_memory=True, persistent_workers=True)

    else:
        raise ValueError(f"无效的dataset_type: {config['dataset_type']}")


    return train_loader, test_loader


def create_model(config):
    """根据配置创建模型实例"""
    # 根据数据集类型确定输入通道数
    if config['dataset_type'] == 'new_multi':
        input_channels = 3
    elif config['dataset_type'] == 'old_multi':
        input_channels = 3
    else:
        raise ValueError(f"无效的dataset_type: {config['dataset_type']}")

    model = TemporalMambaFusionNet(input_nc=3, output_nc=3, base=48).to(config['device'])

    # 加载预训练权重（如果存在）
    pretrained_path = config.get('pretrained_path')
    if pretrained_path and os.path.exists(pretrained_path):
        try:
            state_dict = torch.load(pretrained_path, map_location=config['device'])
            model.load_state_dict(state_dict)
            print(f"✅ 成功加载预训练权重: {pretrained_path}")
        except Exception as e:
            print(f"⚠️ 加载预训练权重失败: {e}")
            print("⚠️ 将从头开始训练")
    else:
        print("ℹ️ 没有提供预训练权重路径或文件不存在，将从头开始训练")

    return model