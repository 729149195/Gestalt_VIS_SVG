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
消融实验 - 变体4：
- 使用标注数据 ✓
- 使用概率选取 ✗
- 使用MDS特征 ✗

这个版本仅保留标注数据的使用，移除概率选取和MDS特征。
主要修改：
1. 移除MDS特征
2. 使用随机采样替代概率选取
3. 保持标注数据的使用
"""

def save_model(args, model, optimizer, scheduler, current_epoch):
    out = os.path.join(args.model_path, f"ablation_no_prob_no_mds_model.tar")
    state = {
        'net': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': current_epoch
    }
    torch.save(state, out)

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.2):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, output1, output2, label):
        output1 = F.normalize(output1, dim=1)
        output2 = F.normalize(output2, dim=1)
        sim = self.cosine_similarity(output1, output2) / self.temperature
        loss = F.binary_cross_entropy_with_logits(sim, label)
        return loss

class Network(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Network, self).__init__()
        self.feature_dim = feature_dim
        self.instance_projector = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, self.feature_dim)
        ) 

    def forward(self, x):
        return self.instance_projector(x)

class FeaturePairDataset(Dataset):
    def __init__(self, data_dir):
        self.sample_pairs = []
        self.load_data(data_dir)

    def get_percentage(self, identifier):
        # 简化的采样比例，不再使用概率选取
        if identifier == 3:
            return 1.0
        elif identifier == 2:
            return 0.75
        elif identifier == 1:
            return 0.5
        else:
            return 1.0

    def sample_pairs_for_one_sample(self, base_feature, candidate_features, identifier, chosen_set, label):
        """
        简化版本的采样函数，使用随机采样替代概率选取
        """
        if len(candidate_features) == 0:
            return []

        selected_pairs = []
        
        if label == 1:  # 正样本对：选择所有样本
            for cf in candidate_features:
                f1_id = self.feature_to_idx[tuple(base_feature.tolist())]
                f2_id = self.feature_to_idx[tuple(cf.tolist())]
                pair_key = (min(f1_id, f2_id), max(f1_id, f2_id), label)
                if pair_key not in chosen_set:
                    chosen_set.add(pair_key)
                    selected_pairs.append((base_feature, cf, label))
        else:  # 负样本对：随机采样
            percentage = self.get_percentage(identifier)
            num_samples = max(1, int(len(candidate_features) * percentage))
            selected_indices = np.random.choice(len(candidate_features), size=num_samples, replace=False)
            
            for idx in selected_indices:
                f1_id = self.feature_to_idx[tuple(base_feature.tolist())]
                f2_id = self.feature_to_idx[tuple(candidate_features[idx].tolist())]
                pair_key = (min(f1_id, f2_id), max(f1_id, f2_id), label)
                if pair_key not in chosen_set:
                    chosen_set.add(pair_key)
                    selected_pairs.append((base_feature, candidate_features[idx], label))

        return selected_pairs

    def load_data(self, data_dir):
        """
        数据加载流程：
        1. 读取所有json文件的all_features和groups
        2. 移除MDS特征
        3. 使用随机采样替代概率选取
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

def train():
    model.train()
    total_loss = 0
    for step, (feature1, feature2, labels) in enumerate(data_loader):
        optimizer.zero_grad()
        feature1 = feature1.to(device)
        feature2 = feature2.to(device)
        labels = labels.to(device).float()
        output1 = model(feature1)
        output2 = model(feature2)
        loss = criterion(output1, output2, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='消融实验：仅使用标注数据版本的对比学习训练')
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
    dataset = FeaturePairDataset(args.data_dir)
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True
    )

    if len(dataset) == 0:
        raise ValueError("数据集中没有样本对，请检查数据集。")

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
        model_fp = os.path.join(args.model_path, "ablation_no_prob_no_mds_model.tar")
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