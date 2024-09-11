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
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


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
    def __init__(self, model_save_path, dataset_path, dataset_path_lr, dataset_path_tb, output_file_mult_path, probabilities_file_path, fourier_file_path,
                 eps, min_samples, distance_threshold_ratio, input_dim=20, feature_dim=20, class_num=20):
        self.model_save_path = model_save_path
        self.dataset_path = dataset_path
        self.dataset_path_lr = dataset_path_lr
        self.dataset_path_tb = dataset_path_tb
        self.output_file_mult_path = output_file_mult_path
        self.probabilities_file_path = probabilities_file_path
        self.fourier_file_path = fourier_file_path
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.class_num = class_num
        self.eps = eps  # DBSCAN eps parameter
        self.min_samples = min_samples  # DBSCAN min_samples parameter
        self.distance_threshold_ratio = distance_threshold_ratio

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ModifiedNetwork(self.input_dim, self.feature_dim, self.class_num).to(self.device)
        self.load_model()

    def load_model(self):
        checkpoint = torch.load(self.model_save_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def predict(self):
        dataset = FeatureVectorDataset(self.dataset_path)
        dateset_lr = FeatureVectorDataset(self.dataset_path_lr)
        dateset_tb = FeatureVectorDataset(self.dataset_path_tb)
        loader = DataLoader(dataset, batch_size=128, shuffle=False)
        loader_lr = DataLoader(dateset_lr, batch_size=128, shuffle=False)
        loader_tb = DataLoader(dateset_tb, batch_size=128, shuffle=False)
        all_identifiers = []
        all_predictions = []
        all_probabilities = []  # For saving cluster probabilities
        all_probabilities_LR = []
        all_probabilities_TB = []
        self.model.eval()

        with torch.no_grad():
            for identifiers, features in loader:
                features = features.to(self.device)
                _, probabilities = self.model(features)
                predicted_clusters = torch.argmax(probabilities, dim=1)
                all_identifiers.extend(identifiers)
                all_predictions.extend(predicted_clusters.tolist())
                all_probabilities.extend(probabilities.tolist())

        with torch.no_grad():
            for identifiers, features in loader_lr:
                features = features.to(self.device)
                _, probabilities = self.model(features)
                predicted_clusters = torch.argmax(probabilities, dim=1)
                all_identifiers.extend(identifiers)
                all_predictions.extend(predicted_clusters.tolist())
                all_probabilities_LR.extend(probabilities.tolist())

        with torch.no_grad():
            for identifiers, features in loader_tb:
                features = features.to(self.device)
                _, probabilities = self.model(features)
                predicted_clusters = torch.argmax(probabilities, dim=1)
                all_identifiers.extend(identifiers)
                all_predictions.extend(predicted_clusters.tolist())
                all_probabilities_TB.extend(probabilities.tolist())


        return all_identifiers, all_predictions, all_probabilities, all_probabilities_LR, all_probabilities_TB

    def compute_fourier_features(self, features):
        # 对每个特征向量进行傅里叶变换，并提取主频率
        fourier_features = []
        for feature in features:
            freqs = fftfreq(len(feature))
            fft_values = fft(feature)
            # 只取实部的模值，并按频率从低到高排序，提取前几个显著频率分量
            magnitude = np.abs(fft_values)
            indices = np.argsort(magnitude)[-15:]  # 提取前15个最显著的频率成分
            fourier_features.append(magnitude[indices])
        return np.array(fourier_features)

    def save_fourier_features_to_json(self, identifiers, fourier_features):
        data = [{"id": identifier, "fourier_features": feature.tolist()} for identifier, feature in
                zip(identifiers, fourier_features)]
        with open(self.fourier_file_path, 'w') as f:
            json.dump(data, f, indent=4)

    def run(self):
        # 获取数据集
        dataset = FeatureVectorDataset(self.dataset_path)
        features = [f.numpy() for _, f in dataset]  # 提取最新的特征向量

        # 使用当前数据进行聚类预测
        identifiers, predicted_clusters, probabilities, probabilities_lr, probabilities_tb = self.predict()
        # 打印当前聚类标签和特征
        # print(f"Current group labels: {predicted_clusters}")
        # print(features)


        # print("features",features)
        # print("probabilities",probabilities)
        probabilities_max = np.maximum.reduce([probabilities,probabilities_lr,probabilities_tb])
        # probabilities = probabilities + probabilities_lr + probabilities_tb

        probabilities_c = np.subtract(probabilities_max, probabilities)
        # print(probabilities_c)

        probabilities_max = probabilities_max.tolist()
        # 保存概率和图矩阵数据
        self.save_probabilities_to_json(identifiers, probabilities_max)

        graph_data = self.generate_graph_data_v2(identifiers, probabilities_max)
        self.save_graph_data_to_json(graph_data)


    def run_distance(self):
        # 获取数据集
        dataset = FeatureVectorDataset(self.dataset_path)
        features = [f.numpy() for _, f in dataset]  # 提取最新的特征向量
        # 使用当前数据进行聚类预测
        identifiers, predicted_clusters, probabilities, probabilities_lr, probabilities_tb = self.predict()
        probabilities_max = np.maximum.reduce([probabilities,probabilities_lr,probabilities_tb])
        probabilities_c = np.subtract(probabilities_max, probabilities)
        # # 打印当前聚类标签和特征
        k_distances_dict = {}
        k_distances = self.calculate_k_distance(probabilities_max, k=3)
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
            plt.plot(sorted_k_distances, label=f'{k}-Distance')

            # 计算一阶差分
            first_diff = np.diff(sorted_k_distances)

            # 使用 find_peaks 来检测曲率较大的点，并计算峰值的显著性(prominence)
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

                # 在图中标记拐点
                color = 'b' if peak == max_prominence_peak else 'r'  # 曲率最大的拐点用蓝色标注，其余用红色
                plt.axvline(x=elbow_index, color=color, linestyle='--')
                plt.text(elbow_index, elbow_distance, f'({elbow_index}, {elbow_distance:.2f})',
                         verticalalignment='bottom', color=color)

        plt.xlabel('Points sorted by distance')
        plt.ylabel('Distance')
        plt.title('K-Distance Plots with Filtered Elbow Points')
        plt.legend()
        plt.grid(True)
        plt.show()

        # 输出所有过滤后的拐点的 distance 值
        # print("All filtered elbow distances:", all_elbow_distances)

        # 输出过滤后曲率最大的拐点的 distance 值
        # print("Max prominence elbow distance:", max_prominence_elbow_distance)
        if max_prominence_elbow_distance is not None:
            return all_elbow_distances, max_prominence_elbow_distance
        else:
            return [0.4], 0.4

    def plot_distances_vs_clusters(self, cluster_counts, intra_class_distances, inter_class_distances):
        plt.figure(figsize=(10, 6))

        # Plotting Intra-class Distance vs Number of Clusters
        plt.plot(cluster_counts, intra_class_distances, label='Average Intra-class Distance', color='blue')

        # Plotting Inter-class Distance vs Number of Clusters
        plt.plot(cluster_counts, inter_class_distances, label='Average Inter-class Distance', color='red')

        # Adding labels and title
        plt.xlabel('Number of Clusters')
        plt.ylabel('Distance')
        plt.title('Intra-class and Inter-class Distances vs. Number of Clusters')
        plt.legend()

        # Show the plot
        plt.grid(True)
        plt.show()

    def calculate_k_distance(self, features, k=2):
        # 转置 features 数据，使每行表示原始列的数据
        features_transposed = np.array(features).T
        # 计算每一行的方差
        variances = np.var(features_transposed, axis=1)
        # 找到方差第二大的行的索引
        sorted_variance_indices = np.argsort(variances)[::-1]  # 从大到小排序
        second_max_variance_idx = sorted_variance_indices[0]  # 第二大方差的行索引

        # 提取方差第二大行对应的特征数据（仅取这一行的数据进行聚类）
        second_max_variance_features = features_transposed[second_max_variance_idx].reshape(-1, 1)


        # 计算每个点到距离它最近的第 k 个点的距离
        dist_matrix = squareform(pdist(features, metric='euclidean'))

        # 对每一行进行排序，并确保只获取有效的距离
        k_distances = []
        for row in dist_matrix:
            sorted_distances = np.sort(row)
            if len(sorted_distances) >= k + 1:
                k_distances.append(sorted_distances[k])
            else:
                # 如果某行距离不足 k+1 个，只取最后一个有效的距离
                k_distances.append(sorted_distances[-1])

        return np.array(k_distances)

    def generate_graph_data_v2(self, identifiers, features):
        graph_data = {
            "GraphData": {
                "node": [],
                "links": [],
                "group": []
            }
        }

        # 转置 features 数据，使每行表示原始列的数据
        features_transposed = np.array(features).T
        # 计算每一行的方差
        variances = np.var(features_transposed, axis=1)
        # 找到方差第二大的行的索引
        sorted_variance_indices = np.argsort(variances)[::-1]  # 从大到小排序
        second_max_variance_idx = sorted_variance_indices[0]  # 第二大方差的行索引

        # 提取方差第二大行对应的特征数据（仅取这一行的数据进行聚类）
        second_max_variance_features = features_transposed[second_max_variance_idx].reshape(-1, 1)

        # 使用DBSCAN进行聚类
        dbscan_groups = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        group_labels = dbscan_groups.fit_predict(features)

        group_dict = {}
        for identifier, group_label in zip(identifiers, group_labels):
            if group_label not in group_dict:
                group_dict[group_label] = []
            group_dict[group_label].append(identifier)

            # 将节点信息加入到node列表中
            graph_data["GraphData"]["node"].append({"id": identifier, "propertyValue": 1})

        graph_data["GraphData"]["group"] = list(group_dict.values())

        def generate_links(all_nodes, all_features, group_info):
            # 计算距离矩阵
            dist_matrix = squareform(pdist(all_features))

            # 计算平均距离
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

        # 遍历所有组，记录每个节点的组别信息
        for group_idx, group in enumerate(graph_data["GraphData"]["group"]):
            indices = [identifiers.index(node) for node in group]
            group_features = np.array(features)[indices]

            # 添加节点和特征到全局集合中
            all_nodes.extend(group)
            all_features.extend(group_features)

            # 记录每个节点的组信息，用于判断是否组内连线
            group_info.extend([group_idx] * len(group))

        # 将特征转换为 numpy 数组
        all_features = np.array(all_features)

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

    def calculate_intra_inter_class_distances(self, features, labels):
        intra_distances = []
        inter_distances = []

        # 将特征和标签分组
        label_to_features = {}
        for feature, label in zip(features, labels):
            if label not in label_to_features:
                label_to_features[label] = []
            label_to_features[label].append(feature)

        # 计算类内距离
        for label, group_features in label_to_features.items():
            if len(group_features) > 1:
                group_features = np.array(group_features)
                dist_matrix = squareform(pdist(group_features, metric='euclidean'))
                intra_distances.extend(dist_matrix[np.triu_indices(len(group_features), k=1)])

        # 计算类间距离
        labels_list = list(label_to_features.keys())
        for i in range(len(labels_list)):
            for j in range(i + 1, len(labels_list)):
                group1 = np.array(label_to_features[labels_list[i]])
                group2 = np.array(label_to_features[labels_list[j]])
                for f1 in group1:
                    for f2 in group2:
                        inter_distances.append(np.linalg.norm(f1 - f2))

        avg_intra_distance = np.mean(intra_distances) if intra_distances else 0
        avg_inter_distance = np.mean(inter_distances) if inter_distances else 0

        return avg_intra_distance, avg_inter_distance



# 运行时指定路径
def main(normalized_csv_path, dataset_path_lr, dataset_path_tb, output_file_mult_path, probabilities_file_path, fourier_file_path, eps=0.4, min_samples=1,
         distance_threshold_ratio=0.3):
    # 创建ClusterPredictor实例
    predictor = ClusterPredictor(
        model_save_path="./static/modules/model_checkpoint_class20_hsl530.tar",
        dataset_path=normalized_csv_path,
        dataset_path_lr = dataset_path_lr,
        dataset_path_tb = dataset_path_tb,
        output_file_mult_path=output_file_mult_path,
        probabilities_file_path=probabilities_file_path,
        fourier_file_path=fourier_file_path,
        eps=eps,
        min_samples=min_samples,
        distance_threshold_ratio=distance_threshold_ratio
    )
    # 首先进行正式的聚类预测和数据保存
    predictor.run()



def get_eps(normalized_csv_path, dataset_path_lr, dataset_path_tb, output_file_mult_path, probabilities_file_path, fourier_file_path, eps=0.4, min_samples=1,
         distance_threshold_ratio=0.3):
    predictor = ClusterPredictor(model_save_path="./static/modules/model_checkpoint_class20_hsl530.tar",
        dataset_path=normalized_csv_path,
        dataset_path_lr=dataset_path_lr,
        dataset_path_tb=dataset_path_tb,
        output_file_mult_path=output_file_mult_path,
        probabilities_file_path=probabilities_file_path,
        fourier_file_path=fourier_file_path,
        eps=eps,
        min_samples=min_samples,
        distance_threshold_ratio=distance_threshold_ratio
    )

    return predictor.run_distance()

    # 然后运行eps范围内的测试，计算和绘制图表
    # predictor.run_with_varying_eps()
