import torch
import numpy as np
import torch.nn as nn


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
        z = nn.functional.normalize(z3, dim=1)
        outputs['normalized_output'] = z
        return z, outputs


model_path = "checkpoint_1000.tar"

# 加载 PyTorch 模型
model = ModifiedNetwork(input_dim=22, feature_dim=4)
checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
model.load_state_dict(checkpoint['net'])
model.eval()

# 提取权重和偏置
linear1 = model.instance_projector[0]
linear2 = model.instance_projector[2]

W1 = linear1.weight.detach().numpy()  # (64, 22)
b1 = linear1.bias.detach().numpy()  # (64,)
W2 = linear2.weight.detach().numpy()  # (4, 64)
b2 = linear2.bias.detach().numpy()  # (4,)

# 计算等效权重和偏置
W_eq = np.dot(W2, W1)  # (4, 22)
b_eq = np.dot(W2, b1) + b2  # (4,)

print("W_eq shape:", W_eq.shape)  # 应输出 (4, 22)
print("b_eq shape:", b_eq.shape)  # 应输出 (4,)


def direct_mapping(input_data, W1, b1, W2, b2):
    """
    直接计算从输入到输出的映射关系，考虑 ReLU 和归一化。

    参数：
    - input_data: 输入数据，形状为 (batch_size, 22)
    - W1: 第一层线性变换的权重，形状为 (64, 22)
    - b1: 第一层线性变换的偏置，形状为 (64,)
    - W2: 第二层线性变换的权重，形状为 (4, 64)
    - b2: 第二层线性变换的偏置，形状为 (4,)

    返回：
    - z: 输出数据，形状为 (batch_size, 4)
    """
    # 线性变换 1
    z1 = np.dot(input_data, W1.T) + b1  # (batch_size, 64)

    # ReLU 激活
    z2 = np.maximum(z1, 0)  # (batch_size, 64)

    # 线性变换 2
    z3 = np.dot(z2, W2.T) + b2  # (batch_size, 4)

    # 归一化
    norm = np.linalg.norm(z3, axis=1, keepdims=True)  # (batch_size, 1)
    z = z3 / norm  # (batch_size, 4)

    return z


# 示例输入
test_inputs = [
    np.random.randn(1, 22).astype(np.float32),
    np.random.randn(5, 22).astype(np.float32),
    np.random.randn(10, 22).astype(np.float32)
]

for i, input_sample in enumerate(test_inputs):
    # PyTorch 模型输出
    input_tensor = torch.tensor(input_sample, dtype=torch.float32)
    with torch.no_grad():
        pytorch_output, _ = model(input_tensor)

    # 直接映射输出
    output_direct = direct_mapping(input_sample, W1, b1, W2, b2)

    # 比较
    difference = np.abs(pytorch_output.numpy() - output_direct)
    max_diff = np.max(difference)

    print(f"测试用例 {i + 1}:")
    print("输入数据:\n", input_sample)
    print("PyTorch 输出:\n", pytorch_output.numpy())
    print("直接映射输出:\n", output_direct)
    print("最大差异:", max_diff)
    print("-" * 50)
