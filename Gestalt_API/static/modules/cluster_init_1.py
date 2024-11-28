import json
import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.nn.functional import normalize
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist, squareform
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import torch.nn.functional as F  # For normalize
import seaborn as sns  # For better visualization

model_path = ("./static/modules/model_checkpoint_class20_llrr_v12.tar")

class ModifiedNetwork(nn.Module):
    def __init__(self, input_dim, feature_dim, class_num):
        super(ModifiedNetwork, self).__init__()
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.cluster_num = class_num

        # Instance projector layers
        self.instance_projector = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.feature_dim)
        )

        # Cluster projector layers
        self.cluster_projector = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.cluster_num),
            nn.Softmax(dim=1)
        )

        self._initialize_weights()

        # Dictionary to store activations
        self.activations = {}

        # Register hooks to capture activations
        self.instance_projector[0].register_forward_hook(self.get_activation('instance_projector.0'))
        self.instance_projector[1].register_forward_hook(self.get_activation('instance_projector.1'))
        self.instance_projector[2].register_forward_hook(self.get_activation('instance_projector.2'))

        self.cluster_projector[0].register_forward_hook(self.get_activation('cluster_projector.0'))
        self.cluster_projector[1].register_forward_hook(self.get_activation('cluster_projector.1'))
        self.cluster_projector[2].register_forward_hook(self.get_activation('cluster_projector.2'))
        # If you want to capture the output of Softmax layer
        # self.cluster_projector[3].register_forward_hook(self.get_activation('cluster_projector.3'))

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def get_activation(self, name):
        def hook(model, input, output):
            self.activations[name] = output.detach()
        return hook

    def forward(self, x):
        # Instance projector forward pass
        z = F.normalize(self.instance_projector(x), dim=1)

        # Cluster projector forward pass
        c = self.cluster_projector(x)

        return z, c

    def forward_cluster(self, x):
        c = self.cluster_projector(x)
        c = torch.argmax(c, dim=1)
        return c

class FeatureVectorDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)

        # 创建副本（identifier结尾为YB_t\YB_b\YB_l\YB_r)
        new_rows = []
        for index, row in self.df.iterrows():
            copyRtoL = row.copy()
            copyRtoL['bbox_left_n'] = row['bbox_right_n']
            copyRtoL['tag_name'] = f"{row['tag_name']}_YB_r"
            new_rows.append(copyRtoL)

            copyLtoR = row.copy()
            copyLtoR['bbox_right_n'] = row['bbox_left_n']
            copyLtoR['tag_name'] = f"{row['tag_name']}_YB_l"
            new_rows.append(copyLtoR)

        # 如果需要将新行添加到原始数据中，可以取消注释以下代码
        # new_df = pd.DataFrame(new_rows)
        # self.df = pd.concat([self.df, new_df], ignore_index=True)

        self.identifiers = self.df.iloc[:, 0].tolist()  # 第一列为标识符
        self.features = self.df.iloc[:, 1:].astype(float).values.tolist()  # 从第二列开始为特征向量

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.identifiers[idx], torch.tensor(self.features[idx], dtype=torch.float32)

class ClusterPredictor:
    def __init__(self, model_save_path, dataset_path, output_file_mult_path, probabilities_file_path,
                 eps, min_samples, distance_threshold_ratio, input_dim=22, feature_dim=22, class_num=20):
        self.model_save_path = model_save_path
        self.dataset_path = dataset_path
        self.output_file_mult_path = output_file_mult_path
        self.probabilities_file_path = probabilities_file_path
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.class_num = class_num
        self.eps = eps  # DBSCAN eps parameter
        self.min_samples = min_samples  # DBSCAN min_samples
        self.distance_threshold_ratio = distance_threshold_ratio
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ModifiedNetwork(self.input_dim, self.feature_dim, self.class_num).to(self.device)
        self.load_model()

    def load_model(self):
        checkpoint = torch.load(self.model_save_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def predict(self):
        dataset = FeatureVectorDataset(self.dataset_path)
        loader = DataLoader(dataset, batch_size=128, shuffle=False)
        all_identifiers = []
        all_probabilities = []
        all_activations = []
        suffixes = ['_YB_t', '_YB_b', '_YB_l', '_YB_r']
        grouped_identifiers = {}
        grouped_activations = {}
        self.model.eval()

        with torch.no_grad():
            for identifiers, features in loader:
                features = features.to(self.device)
                _, probabilities = self.model(features)
                # Copy the activations
                activations = {k: v.cpu() for k, v in self.model.activations.items()}

                for idx, (identifier, prob) in enumerate(zip(identifiers, probabilities.tolist())):
                    base_identifier = identifier
                    for suffix in suffixes:
                        if identifier.endswith(suffix):  # 如果标识符以某个后缀结尾
                            base_identifier = identifier[:-len(suffix)]  # 去掉后缀，得到基础标识符
                            break
                    if base_identifier not in grouped_identifiers:  # 按基础标识符分组存储
                        grouped_identifiers[base_identifier] = []
                        grouped_activations[base_identifier] = []
                    grouped_identifiers[base_identifier].append(prob)  # 将概率加入相应的组
                    # Store activations for this sample
                    sample_activations = {layer: activations[layer][idx].numpy() for layer in activations}
                    grouped_activations[base_identifier].append(sample_activations)

        # 后处理：合并概率，并保留无后缀的基础标识符
        for base_identifier in grouped_identifiers:
            probs = grouped_identifiers[base_identifier]
            activations_list = grouped_activations[base_identifier]

            if len(probs) > 1:  # 如果该基础标识符对应多个概率（即有后缀的标识符）
                max_prob = [max(p) for p in zip(*probs)]  # 逐元素取每个维度的最大值
                # Average activations across duplicates
                avg_activations = {}
                for layer in activations_list[0]:
                    avg_activations[layer] = np.mean(
                        [act[layer] for act in activations_list], axis=0)
            else:
                max_prob = probs[0]
                avg_activations = activations_list[0]

            all_identifiers.append(base_identifier)  # 只保留无后缀的基础标识符
            all_probabilities.append(max_prob)  # 添加合并后的最大概率
            all_activations.append(avg_activations)

        return all_identifiers, all_probabilities, all_activations

    def run(self):
        identifiers, probabilities, activations = self.predict()  # 使用当前数据进行聚类预测
        self.save_probabilities_to_json(identifiers, probabilities)  # 保存概率和图矩阵数据
        graph_data = self.generate_graph_data_v2(identifiers, probabilities)
        self.save_graph_data_to_json(graph_data)
        # 可视化激活
        self.visualize_activations(identifiers, activations)
        # 计算并可视化输入特征的影响
        input_influences = self.compute_input_influence()
        self.visualize_input_influence(input_influences)

    def run_distance(self):
        identifiers, probabilities, _ = self.predict()  # 使用当前数据进行聚类预测
        k_distances_dict = {}
        k_distances = self.calculate_k_distance(probabilities, k=1)
        k_distances_dict[3] = k_distances
        return self.plot_g_distances(k_distances_dict)

    def plot_g_distances(self, k_distances_dict, prominence_factor=0.02, min_distance_diff=0.01):
        all_elbow_distances = []
        max_prominence_elbow_distance = None
        max_prominence_peak = None
        filtered_peaks = []
        max_prominence = -np.inf
        for k, k_distances in k_distances_dict.items():
            sorted_k_distances = np.sort(k_distances)[::-1]  # 从大到小排序
            # plt.plot(sorted_k_distances, label=f'{k}-Distance')
            first_diff = np.diff(sorted_k_distances)  # 计算一阶差分
            peaks, properties = find_peaks(-first_diff, prominence=np.max(-first_diff) * prominence_factor)
            for i, peak in enumerate(peaks):
                elbow_index = peak + 1  # +1 是因为我们取的是差分后的数组
                elbow_distance = sorted_k_distances[elbow_index]
                # 检查当前拐点与前一个拐点的 distance 差值
                if i > 0:
                    prev_elbow_index = peaks[i - 1] + 1
                    prev_elbow_distance = sorted_k_distances[prev_elbow_index]
                    if abs(elbow_distance - prev_elbow_distance) <= min_distance_diff:
                        # 只保留显著性更高的拐点
                        if properties['prominences'][i] > properties['prominences'][i - 1]:
                            filtered_peaks[-1] = peak  # 替换为显著性更高的拐点
                        continue
                filtered_peaks.append(peak)

            # 遍历并标记最终过滤后的拐点
            for peak in filtered_peaks:
                elbow_index = peak + 1  # +1 是因为我们取的是差分后的数组
                elbow_distance = sorted_k_distances[elbow_index]
                all_elbow_distances.append(elbow_distance)

                # 更新最大显著性的拐点（曲率最大）
                if properties['prominences'][filtered_peaks.index(peak)] > max_prominence:
                    max_prominence = properties['prominences'][filtered_peaks.index(peak)]
                    max_prominence_elbow_distance = elbow_distance
                    max_prominence_peak = peak  # 更新最大显著性拐点的索引

        if max_prominence_elbow_distance is not None:
            return all_elbow_distances, max_prominence_elbow_distance
        else:
            return [0.4], 0.4

    def calculate_k_distance(self, features, k):
        # 计算每个点到距离它最近的第 k 个点的距离
        dist_matrix = squareform(pdist(features, metric='euclidean'))
        k_distances = []
        for row in dist_matrix:
            sorted_distances = np.sort(row)
            if len(sorted_distances) >= k + 1:
                k_distances.append(sorted_distances[k])
            else:
                k_distances.append(sorted_distances[-1])  # 如果某行距离不足 k+1 个，只取最后一个有效的距离

        return np.array(k_distances)

    def generate_graph_data_v2(self, identifiers, features):
        graph_data = {
            "GraphData": {
                "node": [],
                "links": [],
                "group": []
            }
        }
        dbscan_groups = DBSCAN(eps=self.eps, min_samples=self.min_samples)  # 使用DBSCAN进行聚类
        group_labels = dbscan_groups.fit_predict(features)

        group_dict = {}
        for identifier, group_label in zip(identifiers, group_labels):
            if group_label not in group_dict:
                group_dict[group_label] = []
            group_dict[group_label].append(identifier)
            graph_data["GraphData"]["node"].append({"id": identifier, "propertyValue": 1})  # 将节点信息加入到node列表中

        graph_data["GraphData"]["group"] = list(group_dict.values())

        def generate_links(all_nodes, all_features, group_info):
            dist_matrix = squareform(pdist(all_features))  # 计算距离矩阵

            mean_distance = np.mean(dist_matrix)  # 计算平均距离
            distance_threshold = mean_distance * self.distance_threshold_ratio
            added_edges = set()
            for i in range(len(all_nodes)):
                for j in range(i + 1, len(all_nodes)):
                    dist = dist_matrix[i, j]
                    if dist < distance_threshold:
                        if (i, j) not in added_edges:
                            # 判断是否是组内连线
                            if group_info[i] == group_info[j]:
                                link_value = dist  # 组内连线保持原值
                            else:
                                link_value = dist / 3  # 非组内连线的value减半
                            graph_data["GraphData"]["links"].append({
                                "source": all_nodes[i],
                                "target": all_nodes[j],
                                "value": link_value
                            })
                            added_edges.add((i, j))

        # 生成所有节点的连线
        all_nodes = []
        all_features = []
        group_info = []
        for group_idx, group in enumerate(graph_data["GraphData"]["group"]):  # 遍历所有组，记录每个节点的组别信息
            indices = [identifiers.index(node) for node in group]
            group_features = np.array(features)[indices]
            all_nodes.extend(group)  # 添加节点和特征到全局集合中
            all_features.extend(group_features)
            group_info.extend([group_idx] * len(group))  # 记录每个节点的组信息，用于判断是否组内连线
        all_features = np.array(all_features)  # 将特征转换为 numpy 数组
        # 生成所有节点之间的连线
        if len(all_nodes) > 1:
            generate_links(all_nodes, all_features, group_info)
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

    def visualize_activations(self, identifiers, activations_list):
        # 可视化每一层的激活
        layers = list(activations_list[0].keys())

        for layer in layers:
            activations = [act[layer] for act in activations_list]
            activations = np.array(activations)

            # 绘制激活的热力图
            plt.figure(figsize=(10, 6))
            sns.heatmap(activations, cmap='viridis')
            plt.title(f'Activations of {layer}')
            plt.xlabel('Neuron Index')
            plt.ylabel('Sample Index')
            plt.show()

    def compute_input_influence(self):
        dataset = FeatureVectorDataset(self.dataset_path)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        input_influences = []
        self.model.eval()

        for identifiers, features in loader:
            features = features.to(self.device)
            features.requires_grad = True  # Enable gradient computation w.r.t. input

            _, probabilities = self.model(features)
            # Let's sum the probabilities to get a scalar output
            output = probabilities.sum()
            self.model.zero_grad()
            output.backward()

            # Gradient w.r.t. input features
            grad = features.grad.detach().cpu().numpy()[0]
            input_influences.append(grad)

        return input_influences

    def visualize_input_influence(self, input_influences):
        input_influences = np.array(input_influences)
        plt.figure(figsize=(10, 6))
        sns.heatmap(input_influences, cmap='coolwarm', center=0)
        plt.title('Input Feature Influence')
        plt.xlabel('Feature Index')
        plt.ylabel('Sample Index')
        plt.show()

# 运行时指定路径
def main(normalized_csv_path, output_file_mult_path, probabilities_file_path, eps=0.4, min_samples=1,
         distance_threshold_ratio=0.3):
    predictor = ClusterPredictor(
        model_save_path=model_path,
        dataset_path=normalized_csv_path,
        output_file_mult_path=output_file_mult_path,
        probabilities_file_path=probabilities_file_path,
        eps=eps,
        min_samples=min_samples,
        distance_threshold_ratio=distance_threshold_ratio
    )
    predictor.run()  # 首先进行正式的聚类预测、数据保存和可视化

def get_eps(normalized_csv_path, output_file_mult_path, probabilities_file_path, eps=0.4, min_samples=1,
            distance_threshold_ratio=0.3):
    predictor = ClusterPredictor(
        model_save_path=model_path,
        dataset_path=normalized_csv_path,
        output_file_mult_path=output_file_mult_path,
        probabilities_file_path=probabilities_file_path,
        eps=eps,
        min_samples=min_samples,
        distance_threshold_ratio=distance_threshold_ratio
    )
    return predictor.run_distance()
