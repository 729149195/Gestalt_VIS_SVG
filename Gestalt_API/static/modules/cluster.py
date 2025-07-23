import json
import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.nn.functional import normalize
import numpy as np

# 模型路径
model_path = ("./static/modules/model_feature_dim_4_batch_64.tar")

# 定义模型类
class ModifiedNetwork(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(ModifiedNetwork, self).__init__()
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        # self.instance_projector = nn.Sequential(
        #     # 第一层：20 -> 32
        #     nn.Linear(input_dim, 32),
        #     nn.BatchNorm1d(32),
        #     nn.ReLU(),
        #     # 第二层：32 -> 16
        #     nn.Linear(32, 16),
        #     nn.BatchNorm1d(16),
        #     nn.ReLU(),
        #     # 第三层：16 -> feature_dim
        #     nn.Linear(16, self.feature_dim),
        #     nn.Tanh()
        # )
        self.instance_projector = nn.Sequential(
            # 第一层：input_dim -> 32（适度扩大特征维度）
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            # 第二层：32 -> 16（平缓降维）
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            # 第三层：16 -> 8（继续降维）
            nn.Linear(16, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            # 最后一层：8 -> 4（最终输出维度）
            nn.Linear(8, self.feature_dim),
            # 保持最终归一化层
            nn.BatchNorm1d(self.feature_dim, affine=False)
        ) 
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        outputs = {}
        x = self.instance_projector[0:3](x)  # 第一层Linear+BN+ReLU
        outputs['layer1_output'] = x
        x = self.instance_projector[3:6](x)  # 第二层Linear+BN+ReLU
        outputs['layer2_output'] = x
        x = self.instance_projector[6:9](x)  # 第三层Linear+BN+ReLU
        outputs['layer3_output'] = x
        x = self.instance_projector[9:](x)   # 最后一层Linear+BN
        z = normalize(x, dim=1)
        outputs['normalized_output'] = z
        return z, outputs

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
                 distance_threshold_ratio, input_dim=20, feature_dim=4):  
        self.model_save_path = model_save_path
        self.dataset_path = dataset_path
        self.output_file_mult_path = output_file_mult_path
        self.features_file_path = features_file_path
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.distance_threshold_ratio = distance_threshold_ratio
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ModifiedNetwork(self.input_dim, self.feature_dim).to(self.device)
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
        data = []
        # print(f"准备序列化数据，特征类型: {type(features)}")
        
        try:
            for i, (identifier, feature) in enumerate(zip(identifiers, features)):
                try:
                    # 更详细地处理各种类型
                    if isinstance(feature, torch.Tensor):
                        # print(f"处理Tensor类型的特征: {i}")
                        feature_list = feature.cpu().numpy().tolist()
                    elif isinstance(feature, np.ndarray):
                        # print(f"处理numpy.ndarray类型的特征: {i}")
                        feature_list = feature.tolist()
                    else:
                        # print(f"处理其他类型的特征: {i}, 类型: {type(feature)}")
                        # 如果是列表，确保其中没有非基本类型
                        if isinstance(feature, list):
                            # 递归检查列表中的每个元素
                            def convert_to_native(item):
                                if isinstance(item, (np.ndarray, torch.Tensor)):
                                    return item.tolist() if hasattr(item, 'tolist') else float(item)
                                elif isinstance(item, list):
                                    return [convert_to_native(subitem) for subitem in item]
                                else:
                                    return item
                            feature_list = convert_to_native(feature)
                        else:
                            feature_list = feature
                    
                    # 确保ID也是可序列化的
                    if isinstance(identifier, (np.ndarray, torch.Tensor)):
                        identifier = identifier.tolist() if hasattr(identifier, 'tolist') else str(identifier)
                    
                    data.append({"id": identifier, "features": feature_list})
                except Exception as e:
                    print(f"处理特征 {i} 时出错: {str(e)}")
                    # 尝试一种更保守的方法
                    if isinstance(feature, (np.ndarray, torch.Tensor)):
                        feature_list = feature.detach().cpu().numpy().tolist() if hasattr(feature, 'detach') else feature.tolist()
                    else:
                        feature_list = str(feature)  # 最后的尝试，转换为字符串
                    data.append({"id": str(identifier), "features": feature_list})
            
            with open(self.features_file_path, 'w') as f:
                json.dump(data, f, indent=4)
                
        except Exception as e:
            print(f"保存特征到JSON文件时出错: {str(e)}")
            print(f"尝试单独序列化每个元素...")
            
            # 尝试序列化每个元素，看哪一个出错
            for i, (identifier, feature) in enumerate(zip(identifiers, features)):
                try:
                    if isinstance(feature, (np.ndarray, torch.Tensor)):
                        feature_list = feature.tolist()
                    else:
                        feature_list = feature
                    json.dumps({"id": identifier, "features": feature_list})
                except Exception as e:
                    print(f"元素 {i} 序列化失败: {str(e)}")
                    print(f"ID类型: {type(identifier)}, 特征类型: {type(feature)}")
                    if isinstance(feature, (np.ndarray, torch.Tensor)):
                        print(f"特征形状: {feature.shape if hasattr(feature, 'shape') else '未知'}")
                    
            # 最后的尝试：全部转换为字符串
            try:
                with open(self.features_file_path, 'w') as f:
                    fallback_data = [{"id": str(identifier), "features": [float(x) for x in feature]} 
                                    for identifier, feature in zip(identifiers, features)]
                    json.dump(fallback_data, f, indent=4)
                print("成功使用后备方法保存JSON")
            except Exception as e:
                print(f"后备方法也失败: {str(e)}")

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
