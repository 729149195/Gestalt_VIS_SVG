import json
import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.nn.functional import normalize
import numpy as np

# 模型路径
model_path = ("./static/modules/best_model_mds211_v2.tar")

# 定义模型类 - 使用PyramidNetwork架构来匹配best_model_mds211_v2.tar
class PyramidNetwork(nn.Module):
    """专为SVG视觉特征优化的金字塔网络架构"""
    def __init__(self, input_dim, feature_dim, dropout_rate=0.2):
        super(PyramidNetwork, self).__init__()
        self.feature_dim = feature_dim
        
        # 第一层 - 22维到16维
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
        z = self.fusion(fusion)
        return z, z

# 定义数据集类
class FeatureVectorDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.identifiers = self.df.iloc[:, 0].tolist()  # 第一列为标识符
        self.features = self.df.iloc[:, 1:].astype(float).values.tolist()  # 从第二列开始为特征向量

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.identifiers[idx], torch.tensor(self.features[idx], dtype=torch.float32)

# 定义聚类预测类
class ClusterPredictor:
    def __init__(self, model_save_path, dataset_path, output_file_mult_path, features_file_path,
                 distance_threshold_ratio, input_dim=22, feature_dim=4):  
        self.model_save_path = model_save_path
        self.dataset_path = dataset_path
        self.output_file_mult_path = output_file_mult_path
        self.features_file_path = features_file_path
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.distance_threshold_ratio = distance_threshold_ratio
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = PyramidNetwork(self.input_dim, self.feature_dim).to(self.device)
        self.load_model()

    def load_model(self):
        checkpoint = torch.load(self.model_save_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['net'])

    # 预测函数
    def predict(self):
        dataset = FeatureVectorDataset(self.dataset_path)
        loader = DataLoader(dataset, batch_size=128, shuffle=False)
        all_identifiers = []
        all_features = []
        self.model.eval()
        with torch.no_grad():
            for identifiers, features in loader:
                features = features.to(self.device)
                z, outputs = self.model(features)
                all_identifiers.extend(identifiers)
                all_features.extend(z.cpu().numpy())
        return all_identifiers, np.array(all_features)

    # 保存特征到 JSON 文件
    def save_features_to_json(self, identifiers, features):
        data = [{"id": identifier, "features": feature.tolist()} for identifier, feature in zip(identifiers, features)]
        with open(self.features_file_path, 'w') as f:
            json.dump(data, f, indent=4)

    def run(self):
        identifiers, features = self.predict()
        
        self.save_features_to_json(identifiers, features)

# 主函数
def main(normalized_csv_path, output_file_mult_path, features_file_path,
         distance_threshold_ratio=0.3):
    predictor = ClusterPredictor(
        model_save_path=model_path,
        dataset_path=normalized_csv_path,
        output_file_mult_path=output_file_mult_path,
        features_file_path=features_file_path,
        distance_threshold_ratio=distance_threshold_ratio
    )
    predictor.run()
