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
    def __init__(self, model_save_path, dataset_path, output_file_mult_path, probabilities_file_path, fourier_file_path,
                 input_dim=20, feature_dim=20, class_num=30, eps=0.4, min_samples=3, distance_threshold_ratio=0.1):
        self.model_save_path = model_save_path
        self.dataset_path = dataset_path
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
        self.model.eval()
        with torch.no_grad():
            for identifiers, features in loader:
                features = features.to(self.device)
                _, probabilities = self.model(features)
                predicted_clusters = torch.argmax(probabilities, dim=1)
                all_identifiers.extend(identifiers)
                all_predictions.extend(predicted_clusters.tolist())
                all_probabilities.extend(probabilities.tolist())
        return all_identifiers, all_predictions, all_probabilities

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
        identifiers, predicted_clusters, probabilities = self.predict()

        # 打印当前聚类标签和特征
        print(f"Current group labels: {predicted_clusters}")

        # 保存概率和图数据
        self.save_probabilities_to_json(identifiers, probabilities)
        graph_data = self.generate_graph_data_v2(identifiers, features)
        self.save_graph_data_to_json(graph_data)

        # 计算3-distance并绘制图表
        # k = 1  # 定义k的值
        #         # k_distances = self.calculate_k_distance(features, k=k)
        #         # self.plot_k_distance(k_distances, k=k)

        # 计算1到5的k-distance并绘制图表
        k_distances_dict = {}
        for k in range(3, 4):
            k_distances = self.calculate_k_distance(features, k=k)
            k_distances_dict[k] = k_distances

        # 绘制k-distance图表
        # self.plot_k_distances(k_distances_dict)
        self.plot_g_distances(k_distances_dict)

    def plot_g_distances(self, k_distances_dict, prominence_factor=0.02, min_distance_diff=0.05):
        plt.figure(figsize=(10, 6))

        for k, k_distances in k_distances_dict.items():
            sorted_k_distances = np.sort(k_distances)[::-1]  # 从大到小排序
            plt.plot(sorted_k_distances, label=f'{k}-Distance')

            # 计算一阶差分
            first_diff = np.diff(sorted_k_distances)

            # 使用 find_peaks 来检测曲率较大的点，并计算峰值的显著性(prominence)
            peaks, properties = find_peaks(-first_diff, prominence=np.max(-first_diff) * prominence_factor)

            filtered_peaks = []
            max_prominence_peak = None
            max_prominence = -np.inf

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

                # 更新最大显著性的拐点（曲率最大）
                if properties['prominences'][i] > max_prominence:
                    max_prominence = properties['prominences'][i]
                    max_prominence_peak = peak

            # 遍历并标记过滤后的拐点
            for peak in filtered_peaks:
                elbow_index = peak + 1  # +1 是因为我们取的是差分后的数组
                elbow_distance = sorted_k_distances[elbow_index]

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

    def run_with_varying_eps(self):
        eps_values = np.arange(0.01, 0.99, 0.01)
        intra_class_distances = []
        inter_class_distances = []
        cluster_counts = []  # 用于记录每个 eps 下的分类数目

        for eps in eps_values:
            self.eps = eps  # Update eps value for each iteration
            print(f"Running with eps = {eps}")

            # 运行 generate_graph_data_v3 来获取距离数据
            dataset = FeatureVectorDataset(self.dataset_path)
            features = [f.numpy() for _, f in dataset]  # 提取最新的特征向量
            identifiers, _, _ = self.predict()
            avg_intra_distance, avg_inter_distance = self.generate_graph_data_v3(identifiers, features)

            # 使用 DBSCAN 获取分类数目
            dbscan_groups = DBSCAN(eps=self.eps, min_samples=self.min_samples)
            group_labels = dbscan_groups.fit_predict(features)
            unique_clusters = len(set(group_labels)) - (1 if -1 in group_labels else 0)  # 排除噪声点
            cluster_counts.append(unique_clusters)

            # Record the distances
            intra_class_distances.append(avg_intra_distance)
            inter_class_distances.append(avg_inter_distance)

        # After looping through all eps values, plot the results
        self.plot_distances_vs_clusters(cluster_counts, intra_class_distances, inter_class_distances)

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

    def calculate_k_distance(self, features, k=3):
        # 计算每个点到距离它最近的第 k 个点的距离
        dist_matrix = squareform(pdist(features, metric='euclidean'))
        k_distances = np.sort(dist_matrix, axis=1)[:, k]  # 排序后取第k个值
        return k_distances


    def plot_k_distances(self, k_distances_dict):
        plt.figure(figsize=(10, 6))
        for k, k_distances in k_distances_dict.items():
            sorted_k_distances = np.sort(k_distances)[::-1]  # 从大到小排序
            plt.plot(sorted_k_distances, label=f'{k}-Distance')
        plt.xlabel('Points sorted by distance')
        plt.ylabel('Distance')
        plt.title('K-Distance Plots')
        plt.legend()
        plt.grid(True)
        plt.show()


    def generate_graph_data_v2(self, identifiers, features):
        graph_data = {
            "GraphData": {
                "node": [],
                "links": [],
                "group": []
            }
        }

        # 使用傅里叶变换提取特征
        fourier_features = self.compute_fourier_features(features)

        # 保存傅里叶特征到JSON文件
        self.save_fourier_features_to_json(identifiers, fourier_features)

        # 使用DBSCAN进行聚类
        dbscan_groups = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        group_labels = dbscan_groups.fit_predict(features)

        # 打印group_labels用于调试
        print("group_labels : ", group_labels)

        group_dict = {}
        for identifier, group_label in zip(identifiers, group_labels):
            if group_label not in group_dict:
                group_dict[group_label] = []
            group_dict[group_label].append(identifier)

            # 将节点信息加入到node列表中
            graph_data["GraphData"]["node"].append({"id": identifier, "propertyValue": 1})

        graph_data["GraphData"]["group"] = list(group_dict.values())

        # 使用MST生成最少连线
        def generate_links(group, group_features):
            if len(group) > 1:
                # 计算距离矩阵
                dist_matrix = squareform(pdist(group_features))

                # 计算平均距离
                mean_distance = np.mean(dist_matrix)

                # 设定距离阈值为平均距离的 50%
                distance_threshold = mean_distance * self.distance_threshold_ratio

                added_edges = set()
                for i in range(len(group)):
                    for j in range(i + 1, len(group)):
                        dist = dist_matrix[i, j]
                        if dist < distance_threshold:
                            if ((i, j) not in added_edges):
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

        # 计算类内距离和类间距离
        avg_intra_distance, avg_inter_distance = self.calculate_intra_inter_class_distances(features, group_labels)
        print(f"Average Intra-class Distance: {avg_intra_distance}")
        print(f"Average Inter-class Distance: {avg_inter_distance}")

        return graph_data

    def generate_graph_data_v3(self, identifiers, features):
        # 使用DBSCAN进行聚类
        dbscan_groups = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        group_labels = dbscan_groups.fit_predict(features)

        # 打印group_labels用于调试
        print("group_labels : ", group_labels)

        # 计算类内距离和类间距离
        avg_intra_distance, avg_inter_distance = self.calculate_intra_inter_class_distances(features, group_labels)
        print(f"Average Intra-class Distance (v3): {avg_intra_distance}")
        print(f"Average Inter-class Distance (v3): {avg_inter_distance}")

        return avg_intra_distance, avg_inter_distance

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
def main(normalized_csv_path, output_file_mult_path, probabilities_file_path, fourier_file_path, eps=0.4, min_samples=3,
         distance_threshold_ratio=0.1):
    # 创建ClusterPredictor实例
    predictor = ClusterPredictor(
        model_save_path="./static/modules/model_checkpoint_class30.tar",
        dataset_path=normalized_csv_path,
        output_file_mult_path=output_file_mult_path,
        probabilities_file_path=probabilities_file_path,
        fourier_file_path=fourier_file_path,
        eps=eps,
        min_samples=min_samples,
        distance_threshold_ratio=distance_threshold_ratio
    )

    # 首先进行正式的聚类预测和数据保存
    predictor.run()

    # 然后运行eps范围内的测试，计算和绘制图表
    # predictor.run_with_varying_eps()

# if __name__ == "__main__":
#     main(
#         normalized_csv_path="../../static/data/normalized_features.csv",
#         output_file_mult_path='../../static/data/community_data_mult.json',
#         probabilities_file_path='../../static/data/cluster_probabilities.json',
#         fourier_file_path='../../static/data/fourier_features.json',
#         eps=0.4,
#         min_samples=3,
#         distance_threshold_ratio = 0.5
#     )
