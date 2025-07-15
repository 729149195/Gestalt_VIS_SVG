# 可视化数据对比学习消融实验

本目录包含了对可视化数据对比学习方法的消融实验实现。我们通过移除不同的组件来研究它们对模型性能的影响。

## 实验设置

我们考虑了三个关键组件：
1. 标注数据的使用 (Annotation)
2. 基于距离的概率选取策略 (Probability)
3. MDS特征 (MDS)

### 实验变体

| 实验ID | 文件名 | Annotation | Probability | MDS | 说明 |
|--------|--------|------------|-------------|-----|------|
| 1      | All_new_instance_only_t-SNE_ZT.py | ✓ | ✓ | ✓ | 完整模型(基准) |
| 2      | train_no_mds.py | ✓ | ✓ | ✗ | 无MDS特征 |
| 3      | train_no_prob.py | ✓ | ✗ | ✓ | 无概率选取 |
| 4      | train_no_prob_no_mds.py | ✓ | ✗ | ✗ | 仅使用标注 |
| 5      | train_unsupervised.py | ✗ | ✓ | ✓ | 无监督+概率+MDS |
| 6      | train_unsupervised_no_mds.py | ✗ | ✓ | ✗ | 无监督+概率 |
| 7      | train_unsupervised_no_prob.py | ✗ | ✗ | ✓ | 无监督+MDS |
| 8      | train_unsupervised_no_prob_no_mds.py | ✗ | ✗ | ✗ | 基础无监督 |

## 运行实验

每个实验变体都可以通过以下方式运行：

```bash
python <实验脚本> [参数]
```

### 通用参数

- `--seed`: 随机种子 (默认: 42)
- `--workers`: 数据加载工作线程数 (默认: 0)
- `--data_dir`: 数据集目录 (默认: ./DataProduce/UpdatedStepGroups_3)
- `--batch_size`: 批大小 (默认: 16)
- `--epochs`: 训练轮数 (默认: 300)
- `--feature_dim`: 特征维度 (默认: 4)
- `--learning_rate`: 学习率 (默认: 0.001)
- `--temperature`: 温度参数 (默认: 0.1)

### 特殊参数

无监督版本额外参数：
- `--distance_threshold`: 距离阈值，用于区分正负样本 (默认: 0.5)

使用概率选取的版本额外参数：
- `--sigma`: 正态分布sigma参数 (默认: 1.0)

## 实验结果保存

所有模型检查点将保存在 `save/ablation_models/` 目录下，文件名格式为：
- 有监督版本：`ablation_[variant]_model.tar`
- 无监督版本：`ablation_unsupervised_[variant]_model.tar`

## 注意事项

1. 无监督版本使用基于距离的启发式方法来确定正负样本对，而不依赖标注数据
2. 移除概率选取的版本使用随机采样替代
3. 移除MDS特征的版本使用较小的输入维度
4. 所有版本保持相同的网络架构和训练策略，只在相应组件上有差异 