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
from scipy.signal import find_peaks

# 模型路径
model_path = ("./static/modules/checkpoint_sort_left.tar")

# 定义模型类
class ModifiedNetwork(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(ModifiedNetwork, self).__init__()
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.instance_projector = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.feature_dim)
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
        z1 = self.instance_projector[0](x)
        outputs['linear1_output'] = z1
        z2 = self.instance_projector[1](z1)
        outputs['relu_output'] = z2
        z3 = self.instance_projector[2](z2)
        outputs['linear2_output'] = z3
        z = normalize(z3, dim=1)
        outputs['normalized_output'] = z
        return z, outputs

# 获取模型权重和偏差的函数
def get_model_weights_biases(model):
    weights_biases = {}
    for name, param in model.named_parameters():
        weights_biases[name] = param.detach().cpu().numpy()
    return weights_biases

# 定义数据集类
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

        # 可以选择是否将新行添加到数据集中
        # new_df = pd.DataFrame(new_rows)
        # self.df = pd.concat([self.df, new_df], ignore_index=True)

        self.identifiers = self.df.iloc[:, 0].tolist()  # 第一列为标识符
        self.features = self.df.iloc[:, 1:].astype(float).values.tolist()  # 从第二列开始为特征向量

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.identifiers[idx], torch.tensor(self.features[idx], dtype=torch.float32)

# 定义聚类预测器类
class ClusterPredictor:
    def __init__(self, model_save_path, dataset_path, output_file_mult_path, features_file_path,
                 eps, min_samples, distance_threshold_ratio, input_dim=21, feature_dim=4):
        self.model_save_path = model_save_path
        self.dataset_path = dataset_path
        self.output_file_mult_path = output_file_mult_path
        self.features_file_path = features_file_path
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.eps = eps  # DBSCAN eps 参数
        self.min_samples = min_samples  # DBSCAN min_samples
        self.distance_threshold_ratio = distance_threshold_ratio
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ModifiedNetwork(self.input_dim, self.feature_dim).to(self.device)
        self.load_model()
        self.subgraphs = {}  # 用于存储每个维度的子图

    def load_model(self):
        checkpoint = torch.load(self.model_save_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['net'])

    # 预测函数
    def predict(self):
        dataset = FeatureVectorDataset(self.dataset_path)
        loader = DataLoader(dataset, batch_size=2048, shuffle=False)
        all_identifiers = []
        all_features = []
        all_layer_outputs = []
        suffixes = ['_YB_t', '_YB_b', '_YB_l', '_YB_r']
        grouped_features = {}
        self.model.eval()
        with torch.no_grad():
            for identifiers, features in loader:
                features = features.to(self.device)
                z, outputs = self.model(features)

                # 对于每个样本
                for idx, identifier in enumerate(identifiers):
                    base_identifier = identifier
                    for suffix in suffixes:
                        if identifier.endswith(suffix):
                            base_identifier = identifier[:-len(suffix)]
                            break
                    if base_identifier not in grouped_features:
                        grouped_features[base_identifier] = []
                    grouped_features[base_identifier].append(z[idx].cpu().numpy())
        # 后处理：合并特征，并保留无后缀的基础标识符
        for base_identifier, features_list in grouped_features.items():
            if len(features_list) > 1:
                max_feature = np.max(features_list, axis=0)
            else:
                max_feature = features_list[0]
            all_identifiers.append(base_identifier)
            all_features.append(max_feature)
        return all_identifiers, all_features

    # 导出模型为 ONNX 文件
    def export_model_to_onnx(self, onnx_file_path='./static/data/model.onnx'):
        self.model.eval()
        dummy_input = torch.randn(1, self.input_dim).to(self.device)
        class ONNXModel(nn.Module):
            def __init__(self, original_model):
                super(ONNXModel, self).__init__()
                self.original_model = original_model
            def forward(self, x):
                z, outputs = self.original_model(x)
                return z  # 仅返回主输出用于 ONNX 导出
        onnx_model = ONNXModel(self.model)
        torch.onnx.export(
            onnx_model,
            dummy_input,
            onnx_file_path,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
            opset_version=11
        )
    # 运行函数
    def run(self):
        identifiers, features = self.predict()
        self.save_features_to_json(identifiers, features)
        graph_data = self.generate_graph_data_v2(identifiers, features)
        self.save_graph_data_to_json(graph_data)
        # 生成每个特征维度的子图
        self.generate_subgraph_for_each_dimension(identifiers, features)


    # 生成每个维度的子图
    def generate_subgraph_for_each_dimension(self, identifiers, features):
        features_array = np.array(features)
        num_nodes = features_array.shape[0]
        num_dimensions = features_array.shape[1]

        for dim in range(num_dimensions):
            dim_features = features_array[:, dim]
            # 计算节点之间的距离矩阵（使用绝对值差异）
            dist_matrix = np.abs(dim_features[:, np.newaxis] - dim_features[np.newaxis, :])
            # 将距离转换为相似度
            similarity_matrix = 1 / (1 + dist_matrix)
            # 构建边列表
            edges = []
            for i in range(num_nodes):
                for j in range(i + 1, num_nodes):
                    sim = similarity_matrix[i, j]
                    edges.append({
                        'source': identifiers[i],
                        'target': identifiers[j],
                        'value': float(sim)
                    })
            # 按相似度从高到低排序边
            edges_sorted = sorted(edges, key=lambda x: x['value'], reverse=True)
            # 仅保留前 20% 的边
            num_edges_to_keep = int(len(edges_sorted) * 0.2)
            edges_filtered = edges_sorted[:num_edges_to_keep]
            graph_data = {
                "nodes": [{"id": id, "name": id} for id in identifiers],
                "links": edges_filtered
            }
            output_dir = os.path.join(os.path.dirname(self.output_file_mult_path), 'subgraphs')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_file = os.path.join(output_dir, f'subgraph_dimension_{dim}.json')
            with open(output_file, 'w') as f:
                json.dump(graph_data, f, indent=4)
            self.subgraphs[dim] = graph_data

    # 生成图数据
    def generate_graph_data_v2(self, identifiers, features):
        graph_data = {
            "GraphData": {
                "node": [],
                "links": [],
                "group": []
            }
        }
        dbscan_groups = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        group_labels = dbscan_groups.fit_predict(features)
        group_dict = {}
        for identifier, group_label in zip(identifiers, group_labels):
            if group_label not in group_dict:
                group_dict[group_label] = []
            group_dict[group_label].append(identifier)
            graph_data["GraphData"]["node"].append({"id": identifier, "propertyValue": 1})
        graph_data["GraphData"]["group"] = list(group_dict.values())
        def generate_links(all_nodes, all_features, group_info):
            dist_matrix = squareform(pdist(all_features))

            mean_distance = np.mean(dist_matrix)
            distance_threshold = mean_distance * self.distance_threshold_ratio
            added_edges = set()
            for i in range(len(all_nodes)):
                for j in range(i + 1, len(all_nodes)):
                    dist = dist_matrix[i, j]
                    if dist < distance_threshold:
                        if (i, j) not in added_edges:
                            # 判断是否是组内连线
                            if group_info[i] == group_info[j]:
                                link_value = dist  # 组内连线
                            else:
                                link_value = dist / 3  # 组间连线
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
        for group_idx, group in enumerate(graph_data["GraphData"]["group"]):
            indices = [identifiers.index(node) for node in group]
            group_features = np.array(features)[indices]
            all_nodes.extend(group)
            all_features.extend(group_features)
            group_info.extend([group_idx] * len(group))
        all_features = np.array(all_features)
        if len(all_nodes) > 1:
            generate_links(all_nodes, all_features, group_info)
        return graph_data

    # 保存图数据到 JSON 文件
    def save_graph_data_to_json(self, graph_data):
        if not os.path.exists(os.path.dirname(self.output_file_mult_path)):
            os.makedirs(os.path.dirname(self.output_file_mult_path))
        with open(self.output_file_mult_path, 'w') as f:
            json.dump(graph_data, f, indent=4)

    # 保存特征到 JSON 文件
    def save_features_to_json(self, identifiers, features):
        data = [{"id": identifier, "features": feature.tolist()} for identifier, feature in zip(identifiers, features)]
        with open(self.features_file_path, 'w') as f:
            json.dump(data, f, indent=4)

    # 运行距离计算
    def run_distance(self):
        identifiers, features = self.predict()
        k_distances_dict = {}
        k_distances = self.calculate_k_distance(features, k=1)
        k_distances_dict[3] = k_distances
        return self.plot_g_distances(k_distances_dict)

    # 绘制距离图
    def plot_g_distances(self, k_distances_dict, prominence_factor=0.02, min_distance_diff=0.01):
        all_elbow_distances = []
        max_prominence_elbow_distance = None
        max_prominence_peak = None
        filtered_peaks = []
        max_prominence = -np.inf
        for k, k_distances in k_distances_dict.items():
            sorted_k_distances = np.sort(k_distances)[::-1]
            first_diff = np.diff(sorted_k_distances)
            peaks, properties = find_peaks(-first_diff, prominence=np.max(-first_diff) * prominence_factor)
            for i, peak in enumerate(peaks):
                elbow_index = peak + 1
                elbow_distance = sorted_k_distances[elbow_index]
                if i > 0:
                    prev_elbow_index = peaks[i - 1] + 1
                    prev_elbow_distance = sorted_k_distances[prev_elbow_index]
                    if abs(elbow_distance - prev_elbow_distance) <= min_distance_diff:
                        if properties['prominences'][i] > properties['prominences'][i - 1]:
                            filtered_peaks[-1] = peak
                        continue
                filtered_peaks.append(peak)

            for peak in filtered_peaks:
                elbow_index = peak + 1
                elbow_distance = sorted_k_distances[elbow_index]
                all_elbow_distances.append(elbow_distance)

                if properties['prominences'][filtered_peaks.index(peak)] > max_prominence:
                    max_prominence = properties['prominences'][filtered_peaks.index(peak)]
                    max_prominence_elbow_distance = elbow_distance
                    max_prominence_peak = peak

        if max_prominence_elbow_distance is not None:
            return all_elbow_distances, max_prominence_elbow_distance
        else:
            return [0.4], 0.4

    # 计算 k-距离
    def calculate_k_distance(self, features, k):
        dist_matrix = squareform(pdist(features, metric='euclidean'))
        k_distances = []
        for row in dist_matrix:
            sorted_distances = np.sort(row)
            if len(sorted_distances) >= k + 1:
                k_distances.append(sorted_distances[k])
            else:
                k_distances.append(sorted_distances[-1])
        return np.array(k_distances)

# 主函数
def main(normalized_csv_path, output_file_mult_path, features_file_path, eps=0.4, min_samples=1, distance_threshold_ratio=0.3):
    predictor = ClusterPredictor(
        model_save_path=model_path,
        dataset_path=normalized_csv_path,
        output_file_mult_path=output_file_mult_path,
        features_file_path=features_file_path,
        eps=eps,
        min_samples=min_samples,
        distance_threshold_ratio=distance_threshold_ratio
    )
    predictor.run()
    predictor.export_model_to_onnx()

# 获取最佳 eps 值
def get_eps(normalized_csv_path, output_file_mult_path, features_file_path, eps=0.4, min_samples=1, distance_threshold_ratio=0.3):
    predictor = ClusterPredictor(
        model_save_path=model_path,
        dataset_path=normalized_csv_path,
        output_file_mult_path=output_file_mult_path,
        features_file_path=features_file_path,
        eps=eps,
        min_samples=min_samples,
        distance_threshold_ratio=distance_threshold_ratio
    )
    return predictor.run_distance()
