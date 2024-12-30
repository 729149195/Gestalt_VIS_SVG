import json
import os
import numpy as np
from sklearn.cluster import DBSCAN
import networkx as nx
import community.community_louvain as community_louvain
from networkx.algorithms.community import label_propagation_communities
from collections import Counter

def load_features_from_json(json_file_path):
    """从JSON文件加载特征数据"""
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    identifiers = []
    features = []
    for item in data:
        identifiers.append(item['id'])
        features.append(item['features'])
    
    return identifiers, np.array(features)

def generate_subgraph(identifiers, features, dimensions, clustering_method):
    """为指定维度生成子图"""
    print(f"\n=== 开始生成子图 ===")
    print(f"使用维度: {dimensions}")
    print(f"聚类方法: {clustering_method}")
    
    selected_features = features[:, dimensions]
    
    # Louvain方法
    if clustering_method.lower() == 'louvain':
        G = nx.Graph()
        for i in range(len(identifiers)):
            G.add_node(identifiers[i])
        
        for i in range(len(identifiers)):
            for j in range(i + 1, len(identifiers)):
                feature_diff = np.linalg.norm(selected_features[i] - selected_features[j])
                similarity = 1 / (1 + feature_diff)
                if similarity > 0.6:
                    G.add_edge(identifiers[i], identifiers[j], weight=similarity)
        
        communities = community_louvain.best_partition(G)
    
    print(f"聚类完成，社区数量: {len(set(communities.values()))}")
    print(f"社区分布: {sorted(Counter(communities.values()).items())}")

    # 构建边列表
    edges = []
    for i in range(len(identifiers)):
        for j in range(i + 1, len(identifiers)):
            if communities[identifiers[i]] == communities[identifiers[j]]:
                feature_diff = np.linalg.norm(selected_features[i] - selected_features[j])
                similarity = 1 / (1 + feature_diff)
                if similarity > 0.4:
                    edges.append({
                        'source': identifiers[i],
                        'target': identifiers[j],
                        'value': float(similarity),
                        'cluster': int(communities[identifiers[i]])
                    })
    
    # 创建图数据
    graph_data = {
        "nodes": [{"id": id, "name": id, "cluster": int(communities[id])} for id in identifiers],
        "links": edges,
        "clusters": len(set(communities.values())),
        "dimensions": dimensions
    }

    return graph_data

def main(features_json_path, output_dir, clustering_method, subgraph_dimensions):
    """主函数"""
    print(f"开始处理子图检测...")
    print(f"使用聚类方法: {clustering_method}")
    print(f"维度组合: {subgraph_dimensions}")

    # 创建输出目录
    subgraphs_dir = os.path.join(output_dir, 'subgraphs')
    if not os.path.exists(subgraphs_dir):
        os.makedirs(subgraphs_dir)

    # 加载特征数据
    identifiers, features = load_features_from_json(features_json_path)

    # 处理每个维度组合
    for dimensions in subgraph_dimensions:
        print(f"\n处理维度组合: {dimensions}")
        
        # 生成子图数据
        graph_data = generate_subgraph(identifiers, features, dimensions, clustering_method)
        
        # 保存子图数据
        dimension_str = ''.join(map(str, dimensions))
        subgraph_file = os.path.join(subgraphs_dir, f'subgraph_dimension_{dimension_str}.json')
        with open(subgraph_file, 'w') as f:
            json.dump(graph_data, f, indent=4)

if __name__ == '__main__':
    # 配置参数
    features_json_path = '../data/cluster_features.json'
    output_dir = '../data'
    clustering_config = {
        'method': 'louvain',
        'dimensions': [
            [0], [1], [2], [3],
            [0,1], [0,2], [0,3],
            [1,2], [1,3], [2,3],
            [0,1,2], [0,1,3], [0,2,3], [1,2,3],
            [0,1,2,3]
        ]
    }
    
    # 运行主函数
    main(
        features_json_path=features_json_path,
        output_dir=output_dir,
        clustering_method=clustering_config['method'],
        subgraph_dimensions=clustering_config['dimensions']
    ) 