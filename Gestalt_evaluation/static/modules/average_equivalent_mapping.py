import torch
import numpy as np
import torch.nn as nn
import pandas as pd
import json
from torch.nn.functional import normalize


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
        return self.fusion(fusion)


class EquivalentWeightsCalculator:
    def __init__(self, model_path, input_dim=22, feature_dim=4):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = PyramidNetwork(input_dim, feature_dim).to(self.device)
        
        # 加载模型权重
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['net'])
            self.model.eval()
            print(f"Successfully loaded model from {model_path}")
        except Exception as e:
            print(f"Warning: Failed to load model from {model_path}: {e}")
            print("Using randomly initialized weights")

    def compute_equivalent_weights(self, input_data, layer1_weights, layer1_bias, layer2_weights, layer2_bias, layer3_weights, layer3_bias, fusion_weights, fusion_bias):
        """
        计算PyramidNetwork的等效权重
        注意：由于PyramidNetwork的复杂结构（包含融合层），等效权重计算会更复杂
        这里提供一个简化的实现
        """
        batch_size = input_data.shape[0]
        
        # 对于PyramidNetwork，由于有融合层，直接计算等效权重会很复杂
        # 这里采用近似方法：使用融合层的权重作为主要等效权重
        
        # 创建一个近似的等效权重矩阵
        W_eq = np.zeros((batch_size, fusion_weights.shape[0], input_data.shape[1]))
        
        for i in range(batch_size):
            # 这是一个简化的等效权重计算
            # 实际的PyramidNetwork等效权重计算会涉及到融合层的复杂结构
            
            # 使用第一层权重作为基础，并通过融合层权重进行加权
            layer1_contribution = np.dot(fusion_weights[:, :16], layer1_weights)
            layer2_contribution = np.dot(fusion_weights[:, 16:28], np.dot(layer2_weights, layer1_weights))
            layer3_contribution = np.dot(fusion_weights[:, 28:], np.dot(layer3_weights, np.dot(layer2_weights, layer1_weights)))
            
            # 组合所有贡献
            W_eq[i] = layer1_contribution + layer2_contribution + layer3_contribution
        
        return W_eq

    def compute_and_save_equivalent_weights(self, csv_file, output_file_avg='average_equivalent_mapping.json', output_file_all='equivalent_weights_by_tag.json'):
        # Read CSV data
        df = pd.read_csv(csv_file)

        # Extract tag_names from the first column
        tag_names = df.iloc[:, 0].astype(str).values  # Ensure they are strings

        # Extract features (assuming features start from the second column)
        input_data = df.iloc[:, 1:].values.astype(np.float32)  # shape: (num_samples, 22)

        # 获取PyramidNetwork的权重和偏置
        layer1_linear = self.model.layer1[0]  # 第一层Linear
        layer2_linear = self.model.layer2[0]  # 第二层Linear  
        layer3_linear = self.model.layer3[0]  # 第三层Linear
        fusion_linear = self.model.fusion[0]  # 融合层Linear

        # 修复CUDA张量转换问题 - 先移动到CPU再转换为numpy
        layer1_weights = layer1_linear.weight.detach().cpu().numpy()  # (16, 22)
        layer1_bias = layer1_linear.bias.detach().cpu().numpy()       # (16,)
        layer2_weights = layer2_linear.weight.detach().cpu().numpy()  # (12, 16)
        layer2_bias = layer2_linear.bias.detach().cpu().numpy()       # (12,)
        layer3_weights = layer3_linear.weight.detach().cpu().numpy()  # (8, 12)
        layer3_bias = layer3_linear.bias.detach().cpu().numpy()       # (8,)
        fusion_weights = fusion_linear.weight.detach().cpu().numpy()  # (4, 36)
        fusion_bias = fusion_linear.bias.detach().cpu().numpy()       # (4,)

        # 计算所有样本的等效权重
        W_eq_all = self.compute_all_equivalent_weights(input_data, layer1_weights, layer1_bias, layer2_weights, layer2_bias, layer3_weights, layer3_bias, fusion_weights, fusion_bias)

        # Compute average equivalent weights
        W_eq_avg = np.mean(W_eq_all, axis=0)  # (feature_dim, 22)

        # Prepare average equivalent weights for JSON
        avg_data = {"average_equivalent_weights": W_eq_avg.tolist()}

        # Save average equivalent weights to JSON
        with open(output_file_avg, 'w') as f:
            json.dump(avg_data, f, indent=4)

        print(f"Average equivalent weights saved to {output_file_avg}")

        # 为每个样本准备等效权重数据
        identifiers = df.iloc[:, 0].tolist()  # 第一列为标识符
        all_data = {}

        for i, identifier in enumerate(identifiers):
            all_data[identifier] = W_eq_all[i].tolist()  # W_eq_all[i] 的形状是 (feature_dim, 22)

        # Save all equivalent weights to JSON
        with open(output_file_all, 'w') as f:
            json.dump(all_data, f, indent=4)

        print(f"All equivalent weights saved to {output_file_all}")

    def compute_all_equivalent_weights(self, input_data, layer1_weights, layer1_bias, layer2_weights, layer2_bias, layer3_weights, layer3_bias, fusion_weights, fusion_bias):
        """
        计算所有样本的等效权重
        """
        num_samples = input_data.shape[0]
        batch_size = 128
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        W_eq_list = []
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)
            batch_inputs = input_data[start_idx:end_idx]
            W_eq_batch = self.compute_equivalent_weights(batch_inputs, layer1_weights, layer1_bias, layer2_weights, layer2_bias, layer3_weights, layer3_bias, fusion_weights, fusion_bias)
            W_eq_list.append(W_eq_batch)
        
        # Concatenate all batches
        return np.concatenate(W_eq_list, axis=0)  # (num_samples, feature_dim, 22)


