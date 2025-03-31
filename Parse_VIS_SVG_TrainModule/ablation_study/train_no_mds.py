import os
import numpy as np
import torch
import json
import argparse
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import math

"""
消融实验 - 变体2：
- 使用标注数据 ✓
- 使用概率选取 ✓
- 使用MDS特征 ✗

这个版本保留了标注数据的使用和概率选取策略，但移除了MDS特征。
主要修改：
1. 输入特征不包含MDS特征
2. 网络结构适应新的输入维度
3. 其他逻辑保持不变
"""

def save_model(args, model, optimizer, scheduler, current_epoch):
    out = os.path.join(args.model_path, f"ablation_no_mds_model.tar")
    state = {
        'net': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': current_epoch
    }
    torch.save(state, out)

# ... [ContrastiveLoss类保持不变] ...

class Network(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Network, self).__init__()
        self.feature_dim = feature_dim
        self.instance_projector = nn.Sequential(
            # 第一层：input_dim -> 32
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            # 第二层：32 -> 16
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            # 第三层：16 -> feature_dim
            nn.Linear(16, self.feature_dim)
        ) 

    def forward(self, x):
        return self.instance_projector(x)

class FeaturePairDataset(Dataset):
    def __init__(self, data_dir, sigma=1.0):
        self.sample_pairs = []
        self.sigma = sigma
        self.load_data(data_dir)

    def bbox_distance(self, f1, f2):
        # 计算定界框距离 - 由于移除MDS特征，需要调整索引
        bbox_f1 = f1[13:17]  # 调整索引以适应无MDS的特征向量
        bbox_f2 = f2[13:17]
        diff = torch.abs(bbox_f1 - bbox_f2)
        dist = diff.min().item()
        return dist

    # ... [其他方法保持不变] ...

    def load_data(self, data_dir):
        """
        数据加载流程：
        1. 读取所有json文件的all_features和groups
        2. 移除MDS特征（最后4个维度）
        3. 其他处理逻辑保持不变
        """
        self.sample_pairs = []
        chosen_set = set()

        for filename in os.listdir(data_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(data_dir, filename)
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    # 移除MDS特征（最后4个维度）
                    all_features = [torch.tensor(feature[:-4], dtype=torch.float32) 
                                  for feature in data['all_features']]
                    self.feature_to_idx = {tuple(f.tolist()): idx for idx, f in enumerate(all_features)}

                    groups = data['groups']
                    for group in groups:
                        if len(group) < 3:
                            continue
                        pos_identifier = group[-2]
                        neg_identifier = group[-1]
                        # 移除MDS特征
                        group_features = [torch.tensor(feature[:-4], dtype=torch.float32) 
                                        for feature in group[:-2]]
                        group_features_set = set(tuple(f.tolist()) for f in group_features)

                        neg_sample_space = [f for f in all_features 
                                          if tuple(f.tolist()) not in group_features_set]

                        # 处理正样本对
                        for i, f_pos in enumerate(group_features):
                            pos_candidates = [f for j,f in enumerate(group_features) if j != i]
                            pos_selected_pairs = self.sample_pairs_for_one_sample(
                                base_feature=f_pos,
                                candidate_features=pos_candidates,
                                identifier=pos_identifier,
                                chosen_set=chosen_set,
                                label=1
                            )
                            self.sample_pairs.extend(pos_selected_pairs)

                        # 处理负样本对
                        for f_pos in group_features:
                            neg_selected_pairs = self.sample_pairs_for_one_sample(
                                base_feature=f_pos,
                                candidate_features=neg_sample_space,
                                identifier=neg_identifier,
                                chosen_set=chosen_set,
                                label=0
                            )
                            self.sample_pairs.extend(neg_selected_pairs)

    def __len__(self):
        return len(self.sample_pairs)

    def __getitem__(self, idx):
        feature1, feature2, label = self.sample_pairs[idx]
        return feature1, feature2, label

# ... [train函数保持不变] ...

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='消融实验：无MDS特征版本的对比学习训练')
    parser.add_argument('--seed', default=42, type=int, help='随机种子')
    parser.add_argument('--workers', default=0, type=int, help='数据加载工作线程数')
    parser.add_argument('--data_dir', default='./DataProduce/UpdatedStepGroups_3', type=str, help='数据集目录')
    parser.add_argument('--batch_size', default=64, type=int, help='批大小')
    parser.add_argument('--start_epoch', default=0, type=int, help='起始epoch')
    parser.add_argument('--epochs', default=300, type=int, help='训练epoch数')
    parser.add_argument('--feature_dim', default=4, type=int, help='特征维度')
    parser.add_argument('--model_path', default='save/ablation_models', type=str, help='模型保存路径')
    parser.add_argument('--reload', action='store_true', help='从检查点重新加载模型')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='学习率')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='权重衰减')
    parser.add_argument('--temperature', default=0.1, type=float, help='温度参数')
    parser.add_argument('--lr_scheduler', default='cosine', type=str, help='学习率调度器类型')
    parser.add_argument('--step_size', default=80, type=int, help='StepLR步长')
    parser.add_argument('--gamma', default=0.1, type=float, help='StepLR衰减率')
    parser.add_argument('--cosine_T_max', default=300, type=int, help='CosineAnnealingLR周期')
    parser.add_argument('--sigma', default=1, type=float, help='正态分布sigma参数')

    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据加载
    dataset = FeaturePairDataset(args.data_dir, sigma=args.sigma)
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True
    )

    if len(dataset) == 0:
        raise ValueError("数据集中没有样本对，请检查数据集。")

    # 获取输入维度（无MDS特征）
    input_dim = dataset.sample_pairs[0][0].size(0)

    # 模型初始化
    model = Network(input_dim, args.feature_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                               lr=args.learning_rate,
                               weight_decay=args.weight_decay)

    # 学习率调度器
    if args.lr_scheduler == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.lr_scheduler == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.cosine_T_max)
    else:
        raise ValueError(f"不支持的学习率调度器类型: {args.lr_scheduler}")

    # 模型加载
    if args.reload:
        model_fp = os.path.join(args.model_path, "ablation_no_mds_model.tar")
        if os.path.exists(model_fp):
            checkpoint = torch.load(model_fp)
            model.load_state_dict(checkpoint['net'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            print(f"从检查点 {model_fp} 重新加载模型，开始训练于 epoch {args.start_epoch}")
        else:
            print(f"检查点文件 {model_fp} 不存在，开始从头训练。")

    criterion = ContrastiveLoss(temperature=args.temperature).to(device)
    
    best_loss = float('inf')
    best_epoch = 0

    # 训练循环
    for epoch in range(args.start_epoch, args.epochs):
        print(f"开始训练 Epoch {epoch + 1}/{args.epochs}")
        loss_epoch = train()
        scheduler.step()
        
        if loss_epoch < best_loss:
            best_loss = loss_epoch
            best_epoch = epoch + 1
            save_model(args, model, optimizer, scheduler, epoch + 1)
            print(f"发现更好的模型！Epoch {epoch + 1}，损失: {loss_epoch:.4f}")
        
        print(f"Epoch [{epoch + 1}/{args.epochs}]\t Loss: {loss_epoch:.4f}\t 最佳Loss: {best_loss:.4f} (Epoch {best_epoch})\t 当前学习率: {optimizer.param_groups[0]['lr']:.6f}")

    print(f"训练完成！最佳模型在 Epoch {best_epoch}，损失为 {best_loss:.4f}") 