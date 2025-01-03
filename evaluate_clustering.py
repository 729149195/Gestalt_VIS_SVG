import os
import json
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import matplotlib.pyplot as plt
import pandas as pd

def load_json_file(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def extract_clusters(data):
    # 分别存储核心聚类和外延聚类的标签
    core_labels = {}
    extension_labels = {}
    
    for cluster in data:
        elements = cluster[:-2]  # 去掉最后两个维度信息
        core_dim, ext_dim = cluster[-2:]  # 获取维度信息
        
        # 为核心维度创建聚类标签
        if core_dim > 0:
            cluster_id = len(core_labels)
            for element in elements:
                core_labels[element] = cluster_id
        
        # 为外延维度创建聚类标签
        if ext_dim > 0:
            cluster_id = len(extension_labels)
            for element in elements:
                extension_labels[element] = cluster_id
    
    return core_labels, extension_labels

def evaluate_clustering(true_labels, pred_labels):
    if not true_labels or not pred_labels:
        return {
            'ARI': None,
            'NMI': None,
            'num_elements': 0
        }
        
    common_elements = set(true_labels.keys()) & set(pred_labels.keys())
    
    if not common_elements:
        return {
            'ARI': None,
            'NMI': None,
            'num_elements': 0
        }
    
    true_list = [true_labels[elem] for elem in common_elements]
    pred_list = [pred_labels[elem] for elem in common_elements]
    
    ari = adjusted_rand_score(true_list, pred_list)
    nmi = normalized_mutual_info_score(true_list, pred_list)
    
    return {
        'ARI': ari,
        'NMI': nmi,
        'num_elements': len(common_elements)
    }

def evaluate_all_steps(model_output_path, ground_truth_dir):
    # 加载模型输出
    model_clusters = load_json_file(model_output_path)
    model_core_labels, model_ext_labels = extract_clusters(model_clusters)
    
    results = {
        'core': {},
        'extension': {}
    }
    
    # 评估每个步骤文件
    for step_file in os.listdir(ground_truth_dir):
        if step_file.startswith('step_') and step_file.endswith('.json'):
            step_path = os.path.join(ground_truth_dir, step_file)
            step_clusters = load_json_file(step_path)
            step_core_labels, step_ext_labels = extract_clusters(step_clusters)
            
            # 评估核心聚类
            core_metrics = evaluate_clustering(step_core_labels, model_core_labels)
            results['core'][step_file] = core_metrics
            
            # 评估外延聚类
            ext_metrics = evaluate_clustering(step_ext_labels, model_ext_labels)
            results['extension'][step_file] = ext_metrics
    
    return results

def visualize_results(results):
    # 创建两个子图，分别显示核心聚类和外延聚类的结果
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    for idx, clustering_type in enumerate(['core', 'extension']):
        df = pd.DataFrame.from_dict(results[clustering_type], orient='index')
        
        # 过滤掉无效值
        valid_ari = df['ARI'].dropna()
        valid_nmi = df['NMI'].dropna()
        
        if len(valid_ari) > 0:
            axes[idx, 0].hist(valid_ari, bins=20)
            axes[idx, 0].set_title(f'{clustering_type.capitalize()} Clustering - ARI Distribution')
            axes[idx, 0].set_xlabel('ARI')
            axes[idx, 0].set_ylabel('Frequency')
            
            axes[idx, 1].hist(valid_nmi, bins=20)
            axes[idx, 1].set_title(f'{clustering_type.capitalize()} Clustering - NMI Distribution')
            axes[idx, 1].set_xlabel('NMI')
            axes[idx, 1].set_ylabel('Frequency')
            
            print(f"\n{clustering_type.capitalize()} Clustering Metrics:")
            print(f"ARI: {valid_ari.mean():.3f} ± {valid_ari.std():.3f}")
            print(f"NMI: {valid_nmi.mean():.3f} ± {valid_nmi.std():.3f}")
            print(f"Number of valid comparisons: {len(valid_ari)}")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 设置路径
    model_output_path = "static/data/subgraphs/subgraph_dimension_all.json"
    ground_truth_dir = "static/data/StepGroups_3"
    
    # 运行评估
    results = evaluate_all_steps(model_output_path, ground_truth_dir)
    
    # 可视化结果
    visualize_results(results) 