import json
import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.nn.functional import normalize
from sklearn.cluster import DBSCAN
from scipy.fft import fft, fftfreq
from scipy.spatial.distance import pdist, squareform
import numpy as np

class ModifiedNetwork(nn.Module):
    def __init__(self, input_dim, feature_dim, class_num):
        super(ModifiedNetwork, self).__init__()
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.cluster_num = class_num
        self.instance_projector = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.feature_dim),
        )
        self.cluster_projector = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.cluster_num),
            nn.Softmax(dim=1)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        z = normalize(self.instance_projector(x), dim=1)
        c = self.cluster_projector(x)
        return z, c

    def forward_cluster(self, x):
        c = self.cluster_projector(x)
        c = torch.argmax(c, dim=1)
        return c


class FeatureVectorDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.identifiers = self.df.iloc[:, 0].tolist()  # 第一列为标识符
        self.features = self.df.iloc[:, 1:].astype(float).values.tolist()  # 从第二列开始为特征向量

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.identifiers[idx], torch.tensor(self.features[idx], dtype=torch.float32)


class ClusterPredictor:
    def __init__(self, model_save_path, dataset_path, output_file_mult_path, probabilities_file_path, fourier_file_path, input_dim=20,
                 feature_dim=20, class_num=30):
        self.model_save_path = model_save_path
        self.dataset_path = dataset_path
        self.output_file_mult_path = output_file_mult_path
        self.probabilities_file_path = probabilities_file_path
        self.fourier_file_path = fourier_file_path
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.class_num = class_num

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ModifiedNetwork(self.input_dim, self.feature_dim, self.class_num).to(self.device)
        self.load_model()

    def load_model(self):
        checkpoint = torch.load(self.model_save_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def predict(self):
        dataset = FeatureVectorDataset(self.dataset_path)
        loader = DataLoader(dataset, batch_size=128, shuffle=False)
        all_identifiers = []
        all_predictions = []
        all_probabilities = []  # For saving cluster probabilities
        top3_indices_list = []  # To store top 3 indices for new functionality
        self.model.eval()
        with torch.no_grad():
            for identifiers, features in loader:
                features = features.to(self.device)
                _, probabilities = self.model(features)
                predicted_clusters = torch.argmax(probabilities, dim=1)
                top3_values, top3_indices = torch.topk(probabilities, k=10, dim=1)
                all_identifiers.extend(identifiers)
                all_predictions.extend(predicted_clusters.tolist())
                all_probabilities.extend(probabilities.tolist())
                top3_indices_list.extend(top3_indices.tolist())
        return all_identifiers, all_predictions, all_probabilities, top3_indices_list

    def compute_fourier_features(self, features):
        # 对每个特征向量进行傅里叶变换，并提取主频率
        fourier_features = []
        for feature in features:
            freqs = fftfreq(len(feature))
            fft_values = fft(feature)
            # 只取实部的模值，并按频率从低到高排序，提取前几个显著频率分量
            magnitude = np.abs(fft_values)
            indices = np.argsort(magnitude)[-15:]  # 提取前20个最显著的频率成分
            fourier_features.append(magnitude[indices])
        return np.array(fourier_features)

    def save_fourier_features_to_json(self, identifiers, fourier_features):
        data = [{"id": identifier, "fourier_features": feature.tolist()} for identifier, feature in zip(identifiers, fourier_features)]
        with open(self.fourier_file_path, 'w') as f:
            json.dump(data, f, indent=4)

    def generate_graph_data_v2(self, identifiers, top3_indices, features):
        graph_data = {
            "GraphData": {
                "node": [],
                "links": [],
                "group": [],
                "subgroups": [],
                "subsubgroups": []
            }
        }

        # # 使用傅里叶变换提取特征
        fourier_features = self.compute_fourier_features(features)
        print("fourier_features", fourier_features)

        # 保存傅里叶特征到JSON文件
        self.save_fourier_features_to_json(identifiers, fourier_features)

        dbscan_Allgroups = DBSCAN(eps=0.4, min_samples=3)

        # dbscan_groups = DBSCAN(eps=0.4, min_samples=3)
        group_labels = dbscan_Allgroups.fit_predict(features)
        print("group_labels : ", group_labels)
        # 初始化group分组
        group_dict = {}

        for identifier, group_label in zip(identifiers, group_labels):
            if group_label not in group_dict:
                group_dict[group_label] = []
            group_dict[group_label].append(identifier)

            # 将节点信息加入到node列表中
            graph_data["GraphData"]["node"].append({"id": identifier, "propertyValue": 1})

        graph_data["GraphData"]["group"] = list(group_dict.values())

        # subgroups的DBSCAN
        # dbscan_subgroups = DBSCAN(eps=0.4, min_samples=1)
        subgroup_dict = {}
        subgroup_labels = dbscan_Allgroups.fit_predict(features)
        print("subgroup_labels : ", subgroup_labels)

        for identifier, subgroup_label in zip(identifiers, subgroup_labels):
            if subgroup_label not in subgroup_dict:
                subgroup_dict[subgroup_label] = []
            subgroup_dict[subgroup_label].append(identifier)

        graph_data["GraphData"]["subgroups"] = list(subgroup_dict.values())

        # subsubgroups的DBSCAN
        # dbscan_subsubgroups = DBSCAN(eps=0.4, min_samples=3)
        subsubgroup_dict = {}
        subsubgroup_labels = dbscan_Allgroups.fit_predict(features)
        print("subsubgroup_labels : ", subsubgroup_labels)

        for identifier, subsubgroup_label in zip(identifiers, subsubgroup_labels):
            if subsubgroup_label not in subsubgroup_dict:
                subsubgroup_dict[subsubgroup_label] = []
            subsubgroup_dict[subsubgroup_label].append(identifier)

        graph_data["GraphData"]["subsubgroups"] = list(subsubgroup_dict.values())

        # 使用MST生成最少连线
        def generate_links(group, group_features, distance_threshold_ratio=0.55):
            if len(group) > 1:
                # 计算距离矩阵
                dist_matrix = squareform(pdist(group_features))

                # 计算平均距离
                mean_distance = np.mean(dist_matrix)

                # 设定距离阈值为平均距离的 50%
                distance_threshold = mean_distance * distance_threshold_ratio

                added_edges = set()
                for i in range(len(group)):
                    for j in range(i + 1, len(group)):
                        dist = dist_matrix[i, j]
                        if dist < distance_threshold:
                            if (i, j) not in added_edges:
                                graph_data["GraphData"]["links"].append({
                                    "source": group[i],
                                    "target": group[j],
                                    "value": dist
                                })
                                added_edges.add((i, j))

            # 生成group内的连线

        for group in graph_data["GraphData"]["group"]:
            indices = [identifiers.index(node) for node in group]
            group_features = np.array(features)[indices]
            if len(group_features) > 1:  # 仅在组内节点数大于1时生成连线
                generate_links(group, group_features)

            # 生成subgroups内的连线
        for group in graph_data["GraphData"]["subgroups"]:
            indices = [identifiers.index(node) for node in group]
            group_features = np.array(features)[indices]
            if len(group_features) > 1:  # 仅在子组节点数大于1时生成连线
                generate_links(group, group_features)

            # 生成subsubgroups内的连线
        for group in graph_data["GraphData"]["subsubgroups"]:
            indices = [identifiers.index(node) for node in group]
            group_features = np.array(features)[indices]
            if len(group_features) > 1:  # 仅在子子组节点数大于1时生成连线
                generate_links(group, group_features)

        return graph_data

    def save_graph_data_to_json(self, graph_data):
        if not os.path.exists(os.path.dirname(self.output_file_mult_path)):
            os.makedirs(os.path.dirname(self.output_file_mult_path))

        with open(self.output_file_mult_path, 'w') as f:
            json.dump(graph_data, f, indent=4)

    def save_probabilities_to_json(self, identifiers, probabilities):
        data = [{"id": identifier, "probabilities": prob} for identifier, prob in zip(identifiers, probabilities)]
        with open(self.probabilities_file_path, 'w') as f:
            json.dump(data, f, indent=4)

    def run(self):
        identifiers, predicted_clusters, probabilities, top3_indices = self.predict()
        features = pd.read_csv(self.dataset_path).iloc[:, 1:].astype(float).values.tolist()
        self.save_probabilities_to_json(identifiers, probabilities)
        graph_data = self.generate_graph_data_v2(identifiers, top3_indices, features)
        self.save_graph_data_to_json(graph_data)


# 运行时指定路径
def main(normalized_csv_path, output_file_mult_path, probabilities_file_path, fourier_file_path):
    predictor = ClusterPredictor(
        model_save_path="./static/modules/model_checkpoint_class30.tar",
        dataset_path=normalized_csv_path,
        output_file_mult_path=output_file_mult_path,
        probabilities_file_path=probabilities_file_path,
        fourier_file_path=fourier_file_path
    )
    predictor.run()


if __name__ == "__main__":
    main(
        normalized_csv_path="../../static/data/normalized_features.csv",
        output_file_mult_path='../../static/data/community_data_mult.json',
        probabilities_file_path='../../static/data/cluster_probabilities.json',
        fourier_file_path='../../static/data/fourier_features.json'
    )
