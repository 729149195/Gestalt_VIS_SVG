import torch
import math
import torch.nn as nn
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import normalize

epochs = 100
temperature = 0.5
batch_size = 128
learning_rate = 0.003
dataset_path = "./normalized_features"
model_save_path = "save/model_checkpoint_self2.tar"  # 设置模型保存路径

# 确保保存路径的目录存在
if not os.path.exists(os.path.dirname(model_save_path)):
    os.makedirs(os.path.dirname(model_save_path))


class InstanceLoss(nn.Module):
    def __init__(self, temperature, device):
        super(InstanceLoss, self).__init__()
        self.temperature = temperature
        self.device = device
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def forward(self, z_i, z_j):
        z = torch.cat([z_i, z_j], dim=0)
        N = z_i.size(0)
        sim = torch.mm(z, z.T) / self.temperature
        sim.fill_diagonal_(-float('inf'))
        labels = torch.arange(N).to(self.device)
        labels = torch.cat([labels, labels], dim=0)
        positives = torch.cat([torch.diag(sim[:N, N:]), torch.diag(sim[N:, :N])], dim=0)
        mask = ~torch.eye(2 * N, dtype=bool).to(self.device)
        negatives = sim[mask].view(2 * N, -1)
        logits = torch.cat((positives.unsqueeze(1), negatives), dim=1)
        labels = torch.zeros(2 * N, dtype=torch.long).to(self.device)
        loss = self.criterion(torch.log_softmax(logits, dim=1), labels)
        return loss.mean()


class DynamicClusterLoss(nn.Module):
    def __init__(self, temperature, device):
        super(DynamicClusterLoss, self).__init__()
        self.temperature = temperature
        self.device = device
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def forward(self, c_i, c_j):
        p_i = c_i.sum(0).view(-1)
        p_i /= p_i.sum() + 1e-8  # 防止除以零
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i + 1e-8)).sum()  # 防止log(0)
        p_j = c_j.sum(0).view(-1)
        p_j /= p_j.sum() + 1e-8  # 防止除以零
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j + 1e-8)).sum()  # 防止log(0)
        ne_loss = ne_i + ne_j
        c_i = c_i.t()
        c_j = c_j.t()
        class_num = c_i.size(0)
        N = 2 * class_num
        c = torch.cat((c_i, c_j), dim=0)
        sim = self.similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag(sim, class_num)
        sim_j_i = torch.diag(sim, -class_num)
        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_clusters = sim.masked_select(~torch.eye(N, dtype=torch.bool).to(self.device)).reshape(N, -1)
        labels = torch.zeros(N).to(self.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(torch.log_softmax(logits, dim=1), labels)
        loss /= N
        return loss + ne_loss


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
    def __init__(self, directory):
        self.directory = directory
        self.files = os.listdir(directory)
        self.features = []
        self.expected_feature_length = 20  # 期望的特征长度
        self.load_features()

    def load_features(self):
        for file in self.files:
            file_path = os.path.join(self.directory, file)
            df = pd.read_csv(file_path)
            for index, row in df.iterrows():
                features = [float(row.iloc[i]) for i in range(1, 21) if not pd.isnull(row.iloc[i])]  # 只选择第2到第21列作为特征
                if len(features) != self.expected_feature_length:
                    print(f"Unexpected feature length in file {file} at row {index}: {len(features)}")
                    continue
                self.features.append(features)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32)


def simplified_train_loop(dataset, model, instance_loss, cluster_loss, optimizer, epochs=epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for features in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            features = features.to(device)
            optimizer.zero_grad()
            z_original, c_original = model(features)
            loss_i = instance_loss(z_original, z_original)  # 使用原始特征训练
            loss_c = cluster_loss(c_original, c_original)  # 使用原始特征训练
            loss = loss_i + loss_c
            if torch.isnan(loss):
                print("Nan loss encountered. Skipping the current batch.")
                print(f"z_original: {z_original}")
                print(f"c_original: {c_original}")
                continue
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(DataLoader(dataset))}")


def save_model(model, optimizer, epoch, save_path):
    state = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_path)


def load_model(model, optimizer, load_path):
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch


if __name__ == "__main__":
    input_dim = 20
    feature_dim = 20
    class_num = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModifiedNetwork(input_dim, feature_dim, class_num).to(device)
    instance_loss = InstanceLoss(temperature=temperature, device=device)
    cluster_loss = DynamicClusterLoss(temperature=temperature, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    dataset = FeatureVectorDataset(dataset_path)
    start_epoch = 0
    if os.path.exists(model_save_path):
        checkpoint = torch.load(model_save_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Loaded checkpoint from epoch {start_epoch}.")
    for epoch in range(start_epoch, epochs):
        total_loss = 0
        for features in DataLoader(dataset, batch_size=batch_size, shuffle=True):
            features = features.to(device)
            optimizer.zero_grad()
            z, c = model(features)
            loss_i = instance_loss(z, z)  # 使用原始特征训练
            loss_c = cluster_loss(c, c)  # 使用原始特征训练
            loss = loss_i + loss_c
            if torch.isnan(loss):
                print("Nan loss encountered. Skipping the current batch.")
                print(f"z: {z}")
                print(f"c: {c}")
                continue
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(DataLoader(dataset))}")
        save_model(model, optimizer, epoch, model_save_path)
