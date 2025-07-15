import os
import numpy as np
import torch
import json
import argparse
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler

def save_model(args, model, optimizer, scheduler, current_epoch):
    out = os.path.join(args.model_path, f"checkpoint_{current_epoch}.tar")
    state = {
        'net': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': current_epoch
    }
    torch.save(state, out)


class ContrastiveLoss(nn.Module):
    """
    带温度参数的对比损失函数。
    接受两个样本的嵌入和目标标签==1（正样本对）或标签==0（负样本对）。
    """
    def __init__(self, temperature=0.2):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, output1, output2, label):
        # 归一化嵌入向量
        output1 = F.normalize(output1, dim=1)
        output2 = F.normalize(output2, dim=1)
        # 计算相似度
        sim = self.cosine_similarity(output1, output2) / self.temperature
        # label已提供正样本对(1)或负样本对(0)
        loss = F.binary_cross_entropy_with_logits(sim, label)
        return loss

class Network(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Network, self).__init__()
        self.feature_dim = feature_dim
        self.instance_projector = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.feature_dim),
        ) 

    def forward(self, x):
        h = x
        z = self.instance_projector(h)
        return z

class FeaturePairDataset(Dataset):
    def __init__(self, data_dir, sigma=1.0):
        self.sample_pairs = []
        self.sigma = sigma  # 高斯核参数
        self.load_data(data_dir)

    def bbox_distance(self, f1, f2):
        # 计算定界框距离，根据指定维度[11:15]
        bbox_f1 = f1[11:15]
        bbox_f2 = f2[11:15]
        diff = torch.abs(bbox_f1 - bbox_f2)
        dist = diff.min().item()
        return dist

    def gaussian_prob(self, distances):
        # 使用高斯核将距离转换为概率分布
        distances = np.array(distances)
        probs = np.exp(-(distances**2)/(2*self.sigma**2))
        if probs.sum() == 0:
            # 避免除0情况，如果全为0（非常极端的情况），则均匀分布
            probs = np.ones_like(probs) / len(probs)
        else:
            probs = probs / probs.sum()
        return probs

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
        对于一个给定基准样本 base_feature，根据 candidate_features 的距离生成高斯分布，按概率抽样。
        与之前的逻辑不同之处在于此处以单个样本为中心。
        """
        if len(candidate_features) == 0:
            return []

        # 计算距离
        distances = []
        for cf in candidate_features:
            dist = self.bbox_distance(base_feature, cf)
            distances.append(dist)

        probs = self.gaussian_prob(distances)
        percentage = self.get_percentage(identifier)
        num_samples = int(len(candidate_features) * percentage)
        if num_samples == 0:
            num_samples = 1

        # 按概率分布进行抽样，避免重复抽样同一元素
        # 首先根据概率从 candidate_features 中随机选择 num_samples 个不重复样本
        selected_indices = np.random.choice(len(candidate_features), size=num_samples, replace=False, p=probs)
        
        selected_pairs = []
        # 获取 base_feature 和 candidate_feature 的独特ID，以避免重复
        # 此处使用 tuple 转为唯一key时需要 feature_to_idx 的辅助
        for idx in selected_indices:
            f1_id = self.feature_to_idx[tuple(base_feature.tolist())]
            f2_id = self.feature_to_idx[tuple(candidate_features[idx].tolist())]
            # 确保pair有序（小的id在前）
            pair_key = (min(f1_id, f2_id), max(f1_id, f2_id), label)
            # 避免重复对
            if pair_key not in chosen_set:
                chosen_set.add(pair_key)
                selected_pairs.append((base_feature, candidate_features[idx], label))
        return selected_pairs

    def load_data(self, data_dir):
        """
        数据加载与处理逻辑:
        1. 从json中读取全部特征 all_features 与 groups。
        2. 构建 feature_to_idx 索引，用于标识特征并避免重复样本对。
        3. 对每一个 group:
           - 分离出 group_features 以及 pos_identifier, neg_identifier。
           - 构建 neg_sample_space (不属于group的特征集合)。
           - 对于正样本对：对 group 内每一个特征为基准，从同组其他特征中通过高斯分布按比例采样出若干正样本对。
           - 对于负样本对：对 group 内每一个特征为基准，从 neg_sample_space 中通过高斯分布按比例采样出若干负样本对。
        4. 去重处理：在采样过程中使用一个全局集合 chosen_set 来确保不加入重复的样本对。
        """
        self.sample_pairs = []
        chosen_set = set()  # 用于记录已选择的样本对，避免重复

        for filename in os.listdir(data_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(data_dir, filename)
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    all_features = [torch.tensor(feature, dtype=torch.float32) for feature in data['all_features']]
                    self.feature_to_idx = {tuple(f.tolist()): idx for idx, f in enumerate(all_features)}

                    groups = data['groups']

                    # 对每个group进行处理
                    for group in groups:
                        if len(group) < 3:
                            continue
                        pos_identifier = group[-2]
                        neg_identifier = group[-1]
                        group_features = [torch.tensor(feature, dtype=torch.float32) for feature in group[:-2]]
                        group_features_set = set(tuple(f.tolist()) for f in group_features)

                        # 构建负样本空间
                        neg_sample_space = [f for f in all_features if tuple(f.tolist()) not in group_features_set]

                        # 正样本对：对每个group内的特征为中心构造分布
                        for i, f_pos in enumerate(group_features):
                            # 候选正样本：同group内除自己之外的特征
                            pos_candidates = [f for idx,f in enumerate(group_features) if idx != i]

                            pos_selected_pairs = self.sample_pairs_for_one_sample(
                                base_feature=f_pos,
                                candidate_features=pos_candidates,
                                identifier=pos_identifier,
                                chosen_set=chosen_set,
                                label=1
                            )
                            self.sample_pairs.extend(pos_selected_pairs)

                        # 负样本对：对每个group内的特征为中心，从neg_sample_space构造分布
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
    # 通用配置
    parser.add_argument('--seed', default=42, type=int, help='随机种子')
    parser.add_argument('--workers', default=0, type=int, help='数据加载工作线程数')
    parser.add_argument('--data_dir', default='./DataProduce/UpdatedStepGroups_3', type=str, help='数据集目录')

    # 训练参数
    parser.add_argument('--batch_size', default=32, type=int, help='批大小')
    parser.add_argument('--start_epoch', default=0, type=int, help='起始epoch')
    parser.add_argument('--epochs', default=300, type=int, help='训练epoch数')

    # 模型参数
    parser.add_argument('--feature_dim', default=4, type=int, help='特征维度')
    parser.add_argument('--model_path', default='save/model_GS', type=str, help='模型保存路径')
    parser.add_argument('--reload', action='store_true', help='从检查点重新加载模型')

    # 损失函数参数
    parser.add_argument('--learning_rate', default=0.01, type=float, help='学习率')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='权重衰减')
    parser.add_argument('--temperature', default=0.2, type=float, help='温度参数')

    # 学习率调度器参数
    parser.add_argument('--lr_scheduler', default='step', type=str, help='学习率调度器类型（例如 "step" 或 "cosine"）')
    parser.add_argument('--step_size', default=100, type=int, help='StepLR 中的 step_size')
    parser.add_argument('--gamma', default=0.1, type=float, help='StepLR 中的 gamma')
    parser.add_argument('--cosine_T_max', default=50, type=int, help='CosineAnnealingLR 中的 T_max')

    # 高斯核 sigma 参数（可选）
    parser.add_argument('--sigma', default=1.0, type=float, help='高斯核sigma参数')

    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 准备数据集
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

    input_dim = dataset.sample_pairs[0][0].size(0)

    # 初始化模型
    model = Network(input_dim, args.feature_dim).to(device)

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.learning_rate,
                                 weight_decay=args.weight_decay)

    # 学习率调度器
    if args.lr_scheduler == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.lr_scheduler == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.cosine_T_max)
    else:
        raise ValueError(f"Unsupported lr_scheduler type: {args.lr_scheduler}")

    # 从检查点加载
    if args.reload:
        model_fp = os.path.join(args.model_path, f"checkpoint_{args.start_epoch}.tar")
        if os.path.exists(model_fp):
            checkpoint = torch.load(model_fp)
            model.load_state_dict(checkpoint['net'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            print(f"从检查点 {model_fp} 重新加载模型，开始训练于 epoch {args.start_epoch}")
        else:
            print(f"检查点文件 {model_fp} 不存在，开始从头训练。")

    # 定义损失函数
    criterion = ContrastiveLoss(temperature=args.temperature).to(device)

    # 训练循环
    for epoch in range(args.start_epoch, args.epochs):
        print(f"开始训练 Epoch {epoch + 1}/{args.epochs}")
        loss_epoch = train()
        scheduler.step()  # 更新学习率
        if (epoch + 1) % 5 == 0:
            save_model(args, model, optimizer, scheduler, epoch + 1)
            print(f"已保存模型至 epoch {epoch + 1}")
        print(f"Epoch [{epoch + 1}/{args.epochs}]\t Loss: {loss_epoch:.4f}\t 当前学习率: {optimizer.param_groups[0]['lr']:.6f}")

    # 保存最终模型
    save_model(args, model, optimizer, scheduler, args.epochs)
    print("训练完成，最终模型已保存。")
