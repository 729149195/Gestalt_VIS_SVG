import json
import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.nn.functional import normalize


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
    def __init__(self, model_save_path, dataset_path, output_file_mult_path, probabilities_file_path, input_dim=20,
                 feature_dim=20, class_num=30):
        self.model_save_path = model_save_path
        self.dataset_path = dataset_path
        self.output_file_mult_path = output_file_mult_path
        self.probabilities_file_path = probabilities_file_path
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.class_num = class_num

        self.model = ModifiedNetwork(self.input_dim, self.feature_dim, self.class_num)
        self.load_model()

    def load_model(self):
        checkpoint = torch.load(self.model_save_path, map_location=torch.device('cpu'))
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
                features = features.to(torch.device('cpu'))
                _, probabilities = self.model(features)
                predicted_clusters = torch.argmax(probabilities, dim=1)
                top3_values, top3_indices = torch.topk(probabilities, k=10, dim=1)
                all_identifiers.extend(identifiers)
                all_predictions.extend(predicted_clusters.tolist())
                all_probabilities.extend(probabilities.tolist())
                top3_indices_list.extend(top3_indices.tolist())
        return all_identifiers, all_predictions, all_probabilities, top3_indices_list

    def save_probabilities_to_json(self, identifiers, probabilities):
        data = [{"id": identifier, "probabilities": prob} for identifier, prob in zip(identifiers, probabilities)]
        with open(self.probabilities_file_path, 'w') as f:
            json.dump(data, f, indent=4)

    def generate_graph_data_v2(self, identifiers, top3_indices):
        graph_data = {
            "GraphData": {
                "node": [],
                "links": [],
                "group": [],
                "subgroups": [],
                "subsubgroups": []
            }
        }

        # Initialize groups to hold nodes based on top3 indices
        groups = {i: [] for i in range(max(max(top3_indices)) + 1)}
        subgroups = {i: [] for i in range(max(max(top3_indices)) + 1)}
        subsubgroups = {i: [] for i in range(max(max(top3_indices)) + 1)}

        for identifier, indices in zip(identifiers, top3_indices):
            # Create node entry
            graph_data["GraphData"]["node"].append({"id": identifier, "propertyValue": 1})

            # Assign node to groups based on top3 indices
            for idx, group_index in enumerate(indices):
                if idx == 0:
                    groups.setdefault(group_index, []).append(identifier)
                elif idx == 1:
                    subgroups.setdefault(group_index, []).append(identifier)
                elif idx == 2:
                    subsubgroups.setdefault(group_index, []).append(identifier)

        # Convert groups to required format and remove empty groups
        graph_data["GraphData"]["group"] = [group for group in [list(set(g)) for g in groups.values()] if group]
        graph_data["GraphData"]["subgroups"] = [group for group in [list(set(g)) for g in subgroups.values()] if group]
        graph_data["GraphData"]["subsubgroups"] = [group for group in [list(set(g)) for g in subsubgroups.values()] if
                                                   group]

        # Generate links within each primary group
        for group in groups.values():
            if len(group) > 1:
                first_node = group[0]
                for other_node in group[1:]:
                    graph_data["GraphData"]["links"].append({
                        "source": first_node,
                        "target": other_node,
                        "value": 1
                    })
        for group in subgroups.values():
            if len(group) > 1:
                first_node = group[0]
                for other_node in group[1:]:
                    graph_data["GraphData"]["links"].append({
                        "source": first_node,
                        "target": other_node,
                        "value": 1
                    })
        for group in subsubgroups.values():
            if len(group) > 1:
                first_node = group[0]
                for other_node in group[1:]:
                    graph_data["GraphData"]["links"].append({
                        "source": first_node,
                        "target": other_node,
                        "value": 1
                    })

        return graph_data

    def save_graph_data_to_json(self, graph_data):
        if not os.path.exists(os.path.dirname(self.output_file_mult_path)):
            os.makedirs(os.path.dirname(self.output_file_mult_path))

        with open(self.output_file_mult_path, 'w') as f:
            json.dump(graph_data, f, indent=4)

    def run(self):
        identifiers, predicted_clusters, probabilities, top3_indices = self.predict()
        self.save_probabilities_to_json(identifiers, probabilities)
        graph_data = self.generate_graph_data_v2(identifiers, top3_indices)
        self.save_graph_data_to_json(graph_data)


# 运行时指定路径
def main(normalized_csv_path, output_file_mult_path, probabilities_file_path):
    predictor = ClusterPredictor(
        model_save_path="./static/modules/model_checkpoint_class30.tar",
        dataset_path=normalized_csv_path,
        output_file_mult_path=output_file_mult_path,
        probabilities_file_path=probabilities_file_path
    )
    predictor.run()


if __name__ == "__main__":
    main(
        normalized_csv_path="../../static/data/normalized_features.csv",
        output_file_mult_path='../../static/data/community_data_mult.json',
        probabilities_file_path='../../static/data/cluster_probabilities.json'
    )
