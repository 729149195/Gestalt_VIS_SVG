import torch
import numpy as np
import onnx
import torch.nn as nn
import onnxruntime as ort
from torch.nn.functional import normalize


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

model_path = ("checkpoint_1000.tar")

# 加载 PyTorch 模型
model = ModifiedNetwork(input_dim=22, feature_dim=4)
checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
model.load_state_dict(checkpoint['net'])
model.eval()

# 加载 ONNX 模型
onnx_model_path = 'model.onnx'
ort_session = ort.InferenceSession(onnx_model_path)

# 测试不同的输入
test_inputs = [
    torch.randn(1, 22),
    torch.randn(5, 22),
    torch.randn(10, 22)
]

for i, test_input in enumerate(test_inputs):
    with torch.no_grad():
        pytorch_output, _ = model(test_input)

    onnx_output = ort_session.run(None, {ort_session.get_inputs()[0].name: test_input.numpy()})[0]

    difference = np.abs(pytorch_output.numpy() - onnx_output)
    max_diff = np.max(difference)

    print(f"测试用例 {i+1}:")
    print("PyTorch 输出:", pytorch_output.numpy())
    print("ONNX 输出:", onnx_output)
    print("最大差异:", max_diff)
    print("-" * 50)
