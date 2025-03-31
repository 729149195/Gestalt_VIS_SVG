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
import time
import re

def save_model(args, model, optimizer, scheduler, current_epoch, loss, is_best=False):
    # 确保模型保存路径存在
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    
    # 创建状态字典，包含更多元数据
    state = {
        'net': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': current_epoch,
        'loss': loss,
        'args': args,
        'timestamp': time.time()
    }
    
    # 保存当前模型
    model_file = os.path.join(args.model_path, f'model_epoch_{current_epoch}.tar')
    torch.save(state, model_file)
    
    # 如果是最佳模型，保存一个副本并保留历史最佳模型
    if is_best:
        best_model_file = os.path.join(args.model_path, 'best_model.tar')
        torch.save(state, best_model_file)
        
        # 保存历史最佳模型（按损失排序，保留top-k）
        history_file = os.path.join(args.model_path, f'best_model_e{current_epoch}_l{loss:.4f}.tar')
        torch.save(state, history_file)
        
        # 保留最近k个最佳模型，删除多余的
        clean_old_models(args.model_path, keep_best=5, keep_latest=3)

def clean_old_models(model_path, keep_best=5, keep_latest=3):
    """保留最近的keep_latest个模型和损失最低的keep_best个模型，删除其他模型"""
    # 获取所有历史最佳模型
    best_models = []
    latest_models = []
    
    for file in os.listdir(model_path):
        if file.startswith("best_model_e") and file.endswith(".tar"):
            # 从文件名解析epoch和损失
            match = re.search(r'best_model_e(\d+)_l([\d\.]+)\.tar', file)
            if match:
                epoch = int(match.group(1))
                loss = float(match.group(2))
                best_models.append((file, loss, epoch))
        elif file.startswith("model_epoch_") and file.endswith(".tar"):
            match = re.search(r'model_epoch_(\d+)\.tar', file)
            if match:
                epoch = int(match.group(1))
                latest_models.append((file, epoch))
    
    # 按损失排序最佳模型并保留top-k
    best_models.sort(key=lambda x: x[1])
    models_to_keep = set([x[0] for x in best_models[:keep_best]])
    
    # 按epoch排序并保留最近k个
    latest_models.sort(key=lambda x: x[1], reverse=True)
    models_to_keep.update([x[0] for x in latest_models[:keep_latest]])
    
    # 删除不在保留列表中的模型
    for file in os.listdir(model_path):
        if (file.startswith("best_model_e") or file.startswith("model_epoch_")) and file.endswith(".tar"):
            if file not in models_to_keep and file != "best_model.tar":
                os.remove(os.path.join(model_path, file))

class WeightedContrastiveLoss(nn.Module):
    """带有样本权重的对比损失"""
    def __init__(self, temperature=0.2, pos_weight=1.0, neg_weight=1.0):
        super(WeightedContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        
    def forward(self, output1, output2, label):
        # 对输出向量进行归一化
        output1 = F.normalize(output1, dim=1)
        output2 = F.normalize(output2, dim=1)
        
        # 计算相似度并缩放
        sim = self.cosine_similarity(output1, output2) / self.temperature
        
        # 根据正负样本给予不同权重
        weights = torch.ones_like(label)
        weights[label == 1] = self.pos_weight
        weights[label == 0] = self.neg_weight
        
        # 使用加权二分类交叉熵损失
        loss = F.binary_cross_entropy_with_logits(sim, label, weight=weights)
        return loss

class PyramidNetwork(nn.Module):
    """专为SVG视觉特征优化的金字塔网络架构"""
    def __init__(self, input_dim, feature_dim, dropout_rate=0.2):
        super(PyramidNetwork, self).__init__()
        self.feature_dim = feature_dim
        
        # 第一层 - 20维到16维
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # 第二层 - 16维到12维
        self.layer2 = nn.Sequential(
            nn.Linear(16, 12),
            nn.BatchNorm1d(12),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # 第三层 - 12维到8维
        self.layer3 = nn.Sequential(
            nn.Linear(12, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # 融合层 - 将所有特征连接并降至目标维度
        self.fusion = nn.Sequential(
            nn.Linear(16 + 12 + 8, feature_dim),
            nn.BatchNorm1d(feature_dim, affine=False)
        )
    
    def forward(self, x):
        # 第一层编码
        x1 = self.layer1(x)
        
        # 第二层编码
        x2 = self.layer2(x1)
        
        # 第三层编码
        x3 = self.layer3(x2)
        
        # 特征融合（连接所有层次的特征）
        fusion = torch.cat([x1, x2, x3], dim=1)
        
        # 输出最终降维结果
        return self.fusion(fusion)

class FeaturePairDataset(Dataset):
    def __init__(self, data_dir, sigma=1.0, perplexity=30.0):
        self.sample_pairs = []
        self.sigma = sigma  # 高斯核参数
        self.perplexity = perplexity  # t-SNE困惑度参数
        self.load_data(data_dir)

    def bbox_distance(self, f1, f2):
        # 计算定界框距离
        bbox_f1 = f1[17:21]
        bbox_f2 = f2[17:21]
        diff = torch.abs(bbox_f1 - bbox_f2)
        dist = diff.min().item()
        return dist
    
    def compute_tsne_probs(self, distances, perplexity=30.0):
        """
        使用t-SNE风格计算条件概率
        距离越近，概率越大
        perplexity控制近邻点的考虑范围
        """
        n = len(distances)
        if n == 0:
            return []
            
        # 转换为numpy数组
        distances = np.array(distances)
        
        # t-SNE中希望每个点的条件概率分布的困惑度等于给定值
        # 困惑度 = 2^熵，所以我们需要找到一个sigma使熵等于log2(perplexity)
        desired_entropy = np.log2(perplexity)
        
        # 初始化sigma搜索范围
        sigma_min = 1e-10
        sigma_max = 1e10
        tol = 1e-5  # 容差
        max_iter = 50  # 最大迭代次数
        
        # 二分搜索找到合适的sigma
        # 简化版：使用固定的sigma
        sigma = self.sigma
        
        # 计算高斯分布下的条件概率
        # p_ij ∝ exp(-d_ij^2 / (2*sigma^2))
        p = np.exp(-distances**2 / (2 * sigma**2))
        
        # 避免自己选自己
        # 在t-SNE中，设置p_ii = 0
        # 这里我们不需要处理，因为我们的距离矩阵不包含自身
        
        # 归一化概率
        p_sum = np.sum(p)
        if p_sum > 0:
            p = p / p_sum
        else:
            # 如果所有点距离都太远，均匀分布
            p = np.ones(n) / n
            
        return p

    def get_percentage(self, identifier):
        # identifier决定最终抽样比例
        if identifier == 3:
            percentage = 1.0
        elif identifier == 2:
            percentage = 0.75
        elif identifier == 1:
            percentage = 0.5
        else:
            percentage = 1.0
        return percentage

    def sample_pairs_for_one_sample(self, base_feature, candidate_features, identifier, chosen_set, label):
        """
        改进的采样策略，使用t-SNE风格的概率选择
        """
        if len(candidate_features) == 0:
            return []

        selected_pairs = []
        
        if label == 1:  # 正样本对：全部保留
            for cf in candidate_features:
                f1_id = self.feature_to_idx[tuple(base_feature.tolist())]
                f2_id = self.feature_to_idx[tuple(cf.tolist())]
                pair_key = (min(f1_id, f2_id), max(f1_id, f2_id), label)
                # 避免重复选择相同的样本对
                if pair_key not in chosen_set:
                    chosen_set.add(pair_key)
                    selected_pairs.append((base_feature, cf, label))
        else:  # 负样本对：使用t-SNE风格概率采样
            distances = []
            for cf in candidate_features:
                dist = self.bbox_distance(base_feature, cf)
                # 避免距离为0的样本
                if dist == 0:
                    dist = 1e-6
                distances.append(dist)

            # 使用t-SNE风格概率分布
            # 距离越近的负样本越难区分，应该有更高的概率被选择
            probs = self.compute_tsne_probs(distances, self.perplexity)
            
            # 确定采样数量
            percentage = self.get_percentage(identifier)
            num_samples = max(1, int(len(candidate_features) * percentage))
            
            # 采样
            selected_indices = np.random.choice(
                len(candidate_features), 
                size=min(num_samples, len(candidate_features)), 
                replace=False, 
                p=probs
            )
            
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
        1. 读取所有json文件的all_features和groups。
        2. 对每个group:
           - 将该group内特征设为group_features。
           - group最后两个值为pos_identifier和neg_identifier。
           - neg_sample_space为所有不在group中的特征。
           - 对每个group内特征f_pos作为中心点，以其余group内特征为正样本候选，neg_sample_space为负样本候选。
           - 使用normal_prob生成概率分布，根据identifier的比例抽样，形成最终的样本对列表。
           - chosen_set用来避免重复样本对。
        """
        self.sample_pairs = []
        chosen_set = set()

        for filename in os.listdir(data_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(data_dir, filename)
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    all_features = [torch.tensor(feature, dtype=torch.float32) for feature in data['all_features']]
                    self.feature_to_idx = {tuple(f.tolist()): idx for idx, f in enumerate(all_features)}

                    groups = data['groups']

                    for group in groups:
                        if len(group) < 3:
                            continue
                        pos_identifier = group[-2]
                        neg_identifier = group[-1]
                        group_features = [torch.tensor(feature, dtype=torch.float32) for feature in group[:-2]]
                        group_features_set = set(tuple(f.tolist()) for f in group_features)

                        neg_sample_space = [f for f in all_features if tuple(f.tolist()) not in group_features_set]

                        # 正样本对选择：每个正样本作为中心
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

                        # 负样本对选择：每个正样本作为中心，从neg_sample_space选取
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
    parser = argparse.ArgumentParser(description='对比学习训练脚本（使用特征向量）')
    parser.add_argument('--seed', default=42, type=int, help='随机种子')
    parser.add_argument('--workers', default=0, type=int, help='数据加载工作线程数')
    parser.add_argument('--data_dir', default='./DataProduce/UpdatedStepGroups_3', type=str, help='数据集目录')

    parser.add_argument('--batch_size', default=64, type=int, help='批大小')
    parser.add_argument('--start_epoch', default=0, type=int, help='起始epoch')
    parser.add_argument('--epochs', default=300, type=int, help='训练epoch数')

    parser.add_argument('--feature_dim', default=4, type=int, help='特征维度')
    parser.add_argument('--model_path', default='save/model_pyramid_v1', type=str, help='模型保存路径')
    parser.add_argument('--reload', action='store_true', help='从检查点重新加载模型')

    parser.add_argument('--learning_rate', default=0.0003, type=float, help='学习率')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='权重衰减')
    parser.add_argument('--temperature', default=0.05, type=float, help='温度参数')

    parser.add_argument('--lr_scheduler', default='cosine', type=str, help='学习率调度器类型（例如 "step" 或 "cosine")')
    parser.add_argument('--step_size', default=80, type=int, help='StepLR 中的 step_size')
    parser.add_argument('--gamma', default=0.1, type=float, help='StepLR 中的 gamma')
    parser.add_argument('--cosine_T_max', default=300, type=int, help='CosineAnnealingLR 中的 T_max')

    parser.add_argument('--sigma', default=1.0, type=float, help='高斯核参数')
    parser.add_argument('--perplexity', default=30.0, type=float, help='t-SNE困惑度参数，控制近邻范围')

    parser.add_argument('--dropout_rate', default=0.2, type=float, help='Dropout比率')
    parser.add_argument('--pos_weight', default=1.2, type=float, help='正样本权重')
    parser.add_argument('--neg_weight', default=0.8, type=float, help='负样本权重')
    parser.add_argument('--patience', default=30, type=int, help='早停耐心值')
    parser.add_argument('--keep_best', default=5, type=int, help='保留的最佳模型数量')
    parser.add_argument('--keep_latest', default=3, type=int, help='保留的最新模型数量')

    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = FeaturePairDataset(args.data_dir, sigma=args.sigma, perplexity=args.perplexity)
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

    # 使用金字塔网络架构
    model = PyramidNetwork(
        input_dim, 
        args.feature_dim,
        dropout_rate=args.dropout_rate
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.learning_rate,
                                 weight_decay=args.weight_decay)

    if args.lr_scheduler == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.lr_scheduler == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.cosine_T_max)
    else:
        raise ValueError(f"Unsupported lr_scheduler type: {args.lr_scheduler}")

    if args.reload:
        model_fp = os.path.join(args.model_path, "best_model.tar")
        if os.path.exists(model_fp):
            checkpoint = torch.load(model_fp)
            model.load_state_dict(checkpoint['net'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            print(f"从检查点 {model_fp} 重新加载模型，开始训练于 epoch {args.start_epoch}")
        else:
            print(f"检查点文件 {model_fp} 不存在，开始从头训练。")

    criterion = WeightedContrastiveLoss(
        temperature=args.temperature,
        pos_weight=args.pos_weight,
        neg_weight=args.neg_weight
    ).to(device)
    
    best_loss = float('inf')
    best_epoch = 0
    patience_counter = 0

    for epoch in range(args.start_epoch, args.epochs):
        print(f"开始训练 Epoch {epoch + 1}/{args.epochs}")
        loss_epoch = train()
        scheduler.step()
        
        if loss_epoch < best_loss:
            best_loss = loss_epoch
            best_epoch = epoch + 1
            patience_counter = 0
            save_model(args, model, optimizer, scheduler, epoch + 1, loss_epoch, is_best=True)
            print(f"发现更好的模型！Epoch {epoch + 1}，损失: {loss_epoch:.4f}")
        else:
            patience_counter += 1
            print(f"没有改进。耐心计数：{patience_counter}/{args.patience}")
            if (epoch + 1) % 10 == 0:
                save_model(args, model, optimizer, scheduler, epoch + 1, loss_epoch, is_best=False)
        
        if patience_counter >= args.patience:
            print(f"早停触发！{args.patience} epoch没有改善。")
            break
            
        print(f"Epoch [{epoch + 1}/{args.epochs}]\t Loss: {loss_epoch:.4f}\t 最佳Loss: {best_loss:.4f} (Epoch {best_epoch})\t 当前学习率: {optimizer.param_groups[0]['lr']:.6f}")

    print(f"训练完成！最佳模型在 Epoch {best_epoch}，损失为 {best_loss:.4f}")
