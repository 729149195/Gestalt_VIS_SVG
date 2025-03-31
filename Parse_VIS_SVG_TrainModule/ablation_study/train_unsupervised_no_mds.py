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
消融实验 - 变体6：
- 使用标注数据 ✗
- 使用概率选取 ✓
- 使用MDS特征 ✗

这个版本是无监督学习版本，保留概率选取但移除MDS特征。
主要修改：
1. 移除MDS特征
2. 保持基于距离的概率选取
3. 使用无监督学习方式
"""

def save_model(args, model, optimizer, scheduler, current_epoch):
    out = os.path.join(args.model_path, f"ablation_unsupervised_no_mds_model.tar")
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

class UnsupervisedFeaturePairDataset(Dataset):
    def __init__(self, data_dir, sigma=1.0, distance_threshold=0.5):
        self.sample_pairs = []
        self.sigma = sigma
        self.distance_threshold = distance_threshold
        self.load_data(data_dir)

    def bbox_distance(self, f1, f2):
        # 计算定界框距离 - 由于移除MDS特征，需要调整索引
        bbox_f1 = f1[13:17]  # 调整索引以适应无MDS的特征向量
        bbox_f2 = f2[13:17]
        diff = torch.abs(bbox_f1 - bbox_f2)
        dist = diff.min().item()
        return dist

    def normal_prob(self, distances):
        # 使用正态分布pdf计算概率
        distances = np.array(distances)
        sigma = self.sigma
        pdf = (1.0 / (sigma * math.sqrt(2*math.pi))) * np.exp(-(distances**2) / (2*sigma*sigma))
        if pdf.sum() == 0:
            pdf = np.ones_like(pdf) / len(pdf)
        else:
            pdf = pdf / pdf.sum()
        return pdf

    def sample_pairs_for_feature(self, base_feature, all_features, chosen_set):
        """
        无监督版本的采样函数：
        - 基于距离阈值确定正负样本
        - 使用概率选取进行采样
        """
        if len(all_features) == 0:
            return []

        selected_pairs = []
        distances = []
        candidates = []
        
        for f in all_features:
            if tuple(f.tolist()) == tuple(base_feature.tolist()):
                continue
            dist = self.bbox_distance(base_feature, f)
            distances.append(dist)
            candidates.append(f)

        if not candidates:
            return []

        # 根据距离阈值划分正负样本
        distances = np.array(distances)
        pos_mask = distances <= self.distance_threshold
        neg_mask = ~pos_mask

        # 处理正样本
        pos_candidates = [f for i, f in enumerate(candidates) if pos_mask[i]]
        if pos_candidates:
            pos_distances = distances[pos_mask]
            pos_probs = self.normal_prob(pos_distances)
            num_pos_samples = max(1, int(len(pos_candidates) * 0.5))  # 采样50%的正样本
            pos_indices = np.random.choice(len(pos_candidates), 
                                         size=min(num_pos_samples, len(pos_candidates)), 
                                         replace=False, p=pos_probs)
            
            for idx in pos_indices:
                f1_id = self.feature_to_idx[tuple(base_feature.tolist())]
                f2_id = self.feature_to_idx[tuple(pos_candidates[idx].tolist())]
                pair_key = (min(f1_id, f2_id), max(f1_id, f2_id), 1)
                if pair_key not in chosen_set:
                    chosen_set.add(pair_key)
                    selected_pairs.append((base_feature, pos_candidates[idx], 1))

        # 处理负样本
        neg_candidates = [f for i, f in enumerate(candidates) if neg_mask[i]]
        if neg_candidates:
            neg_distances = distances[neg_mask]
            neg_probs = self.normal_prob(neg_distances)
            num_neg_samples = max(1, int(len(neg_candidates) * 0.3))  # 采样30%的负样本
            neg_indices = np.random.choice(len(neg_candidates), 
                                         size=min(num_neg_samples, len(neg_candidates)), 
                                         replace=False, p=neg_probs)
            
            for idx in neg_indices:
                f1_id = self.feature_to_idx[tuple(base_feature.tolist())]
                f2_id = self.feature_to_idx[tuple(neg_candidates[idx].tolist())]
                pair_key = (min(f1_id, f2_id), max(f1_id, f2_id), 0)
                if pair_key not in chosen_set:
                    chosen_set.add(pair_key)
                    selected_pairs.append((base_feature, neg_candidates[idx], 0))

        return selected_pairs

    def load_data(self, data_dir):
        """
        无监督数据加载流程：
        1. 只读取all_features，不使用groups
        2. 移除MDS特征
        3. 对每个特征，基于距离选择正负样本
        4. 使用概率选取进行采样
        """
        self.sample_pairs = []
        chosen_set = set()
        all_features_list = []

        # 首先收集所有特征
        for filename in os.listdir(data_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(data_dir, filename)
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    # 移除MDS特征（最后4个维度）
                    features = [torch.tensor(feature[:-4], dtype=torch.float32) 
                              for feature in data['all_features']]
                    all_features_list.extend(features)

        # 创建特征到索引的映射
        self.feature_to_idx = {tuple(f.tolist()): idx for idx, f in enumerate(all_features_list)}

        # 为每个特征生成样本对
        for base_feature in all_features_list:
            other_features = [f for f in all_features_list 
                            if tuple(f.tolist()) != tuple(base_feature.tolist())]
            pairs = self.sample_pairs_for_feature(base_feature, other_features, chosen_set)
            self.sample_pairs.extend(pairs)

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
    parser = argparse.ArgumentParser(description='消融实验：无监督+概率选取但无MDS特征版本的对比学习训练')
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
    parser.add_argument('--sigma', default=1.0, type=float, help='正态分布sigma参数')
    parser.add_argument('--distance_threshold', default=0.5, type=float, help='距离阈值，用于区分正负样本')

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
    dataset = UnsupervisedFeaturePairDataset(
        args.data_dir, 
        sigma=args.sigma,
        distance_threshold=args.distance_threshold
    )
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
        model_fp = os.path.join(args.model_path, "ablation_unsupervised_no_mds_model.tar")
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