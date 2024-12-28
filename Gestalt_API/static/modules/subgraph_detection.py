import json
import os
import numpy as np
from sklearn.cluster import DBSCAN, MeanShift, SpectralClustering
import networkx as nx
import community.community_louvain as community_louvain
from networkx.algorithms.community import label_propagation_communities
from networkx.algorithms.community import girvan_newman
from collections import Counter
from infomap import Infomap
import igraph as ig
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial import distance_matrix

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

def detect_grid_structure(identifiers, features, dimensions):
    """检测SVG元素中的网格结构"""
    print("开始检测网格结构...")
    selected_features = np.array(features)[:, dimensions]
    n_dims = selected_features.shape[1]
    print(f"当前处理{n_dims}维数据")

    # 1. 基于位置的网格检测
    def detect_position_grid(positions):
        clusters_per_dim = []
        for dim in range(positions.shape[1]):
            coords = positions[:, dim]
            clustering = MeanShift(bandwidth=0.05).fit(coords.reshape(-1, 1))
            clusters_per_dim.append(clustering.labels_)

        grid_cells = {}
        for i in range(len(identifiers)):
            cell_key = ",".join(str(clusters_per_dim[dim][i]) for dim in range(n_dims))
            if cell_key not in grid_cells:
                grid_cells[cell_key] = []
            grid_cells[cell_key].append(identifiers[i])

        return grid_cells

    # 2. 基于层次聚类的网格检测
    def detect_hierarchical_grid(positions):
        dist_matrix = distance_matrix(positions, positions)
        linkage_matrix = linkage(dist_matrix, method='ward')
        threshold = 0.1 * np.sqrt(n_dims)
        clusters = fcluster(linkage_matrix, threshold, criterion='distance')
        
        grid_clusters = {}
        for i, cluster_id in enumerate(clusters):
            cluster_key = str(cluster_id)
            if cluster_key not in grid_clusters:
                grid_clusters[cluster_key] = []
            grid_clusters[cluster_key].append(identifiers[i])
        
        return grid_clusters

    # 3. 基于对齐的网格��测
    def detect_alignment_grid(positions):
        def find_aligned_elements(coords, tolerance=0.02):
            sorted_indices = np.argsort(coords)
            aligned_groups = []
            current_group = [sorted_indices[0]]
            
            for i in range(1, len(sorted_indices)):
                if abs(coords[sorted_indices[i]] - coords[sorted_indices[i-1]]) < tolerance:
                    current_group.append(sorted_indices[i])
                else:
                    if len(current_group) > 1:
                        aligned_groups.append(current_group)
                    current_group = [sorted_indices[i]]
            
            if len(current_group) > 1:
                aligned_groups.append(current_group)
            
            return aligned_groups

        alignment_groups = {}
        for dim in range(n_dims):
            dim_coords = positions[:, dim]
            aligned_groups = find_aligned_elements(dim_coords)
            alignment_groups[f'dim_{dim}'] = [
                [identifiers[idx] for idx in group] 
                for group in aligned_groups
            ]

        return alignment_groups

    grid_structures = {
        'position_grid': detect_position_grid(selected_features),
        'hierarchical_grid': detect_hierarchical_grid(selected_features),
        'alignment_grid': detect_alignment_grid(selected_features),
        'dimensions_info': {
            'n_dimensions': n_dims,
            'dimension_indices': dimensions
        }
    }

    return grid_structures

def generate_subgraph(identifiers, features, dimensions, clustering_method):
    """为指定维度生成子图"""
    print(f"\n=== 开始生成子图 ===")
    print(f"使用维度: {dimensions}")
    print(f"聚类方法: {clustering_method}")
    
    selected_features = features[:, dimensions]
    
    # DBSCAN方法
    if clustering_method.lower() == 'dbscan':
        eps = float(np.mean(np.std(selected_features, axis=0))) * 0.35
        min_samples = max(3, int(len(selected_features) * 0.03))
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(selected_features)
        communities = {id: label for id, label in zip(identifiers, clustering.labels_)}
    
    # Louvain方法
    elif clustering_method.lower() == 'louvain':
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
    
    # Label Propagation方法
    elif clustering_method.lower() == 'label_propagation':
        G = nx.Graph()
        for i in range(len(identifiers)):
            G.add_node(identifiers[i])
        
        for i in range(len(identifiers)):
            for j in range(i + 1, len(identifiers)):
                feature_diff = np.linalg.norm(selected_features[i] - selected_features[j])
                similarity = 1 / (1 + feature_diff)
                if similarity > 0.4:
                    G.add_edge(identifiers[i], identifiers[j], weight=similarity)
        
        communities_generator = label_propagation_communities(G)
        communities_list = list(communities_generator)
        communities = {}
        for i, community in enumerate(communities_list):
            for node in community:
                communities[node] = i
    
    # 其他聚类方法的实现...
    else:
        raise ValueError(f"不支持的聚类方法: {clustering_method}")

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

    # 如果有至少2个维度，添加网格检测
    if len(dimensions) >= 2:
        grid_structures = detect_grid_structure(identifiers, features, dimensions)
        print(f"网格检测完成，发现 {sum(len(cells) for cells in grid_structures['position_grid'].values())} 个网格单元")
        graph_data['grid_structures'] = grid_structures

    return graph_data

def main(features_json_path, output_dir, clustering_method, subgraph_dimensions):
    """主函数"""
    print(f"开始处理子图检测...")
    print(f"使用聚类方法: {clustering_method}")
    print(f"维度组合: {subgraph_dimensions}")

    # 创建输出目录
    subgraphs_dir = os.path.join(output_dir, 'subgraphs')
    grid_structures_dir = os.path.join(output_dir, 'grid_structures')
    for dir_path in [subgraphs_dir, grid_structures_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

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
        
        # 如果有网格结构数据，单独保存
        if 'grid_structures' in graph_data:
            grid_file = os.path.join(grid_structures_dir, f'grid_detection_dimension_{dimension_str}.json')
            with open(grid_file, 'w') as f:
                json.dump(graph_data['grid_structures'], f, indent=4)

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