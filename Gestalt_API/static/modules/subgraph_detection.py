import json
import os
import numpy as np
from sklearn.cluster import DBSCAN
import networkx as nx
import community.community_louvain as community_louvain
from networkx.algorithms.community import label_propagation_communities
from collections import Counter, defaultdict

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

def analyze_cluster_overlaps(subgraphs_dir):
    """分析所有子图中聚类的重叠情况，生成核心聚类和外延"""
    print("\n=== 开始分析聚类重叠 ===")
    
    # 第一步：收集所有维度的聚类，每个聚类都视为核心聚类
    core_clusters = []
    
    for filename in os.listdir(subgraphs_dir):
        if filename.startswith('subgraph_dimension_') and filename != 'subgraph_dimension_all.json':
            dimension = filename.replace('subgraph_dimension_', '').replace('.json', '')
            with open(os.path.join(subgraphs_dir, filename), 'r') as f:
                graph_data = json.load(f)
                
                # 收集该维度下的所有聚类
                clusters = defaultdict(set)
                for node in graph_data['nodes']:
                    clusters[node['cluster']].add(node['id'])
                
                # 将每个聚类作为一个核心聚类
                for cluster_id, nodes in clusters.items():
                    # 转换维度标识 (dimension + 1)
                    dimension_str = str(int(dimension) + 1) if len(dimension) == 1 else ''.join(str(int(d) + 1) for d in dimension)
                    core_clusters.append({
                        'core_nodes': list(nodes),
                        'core_dimensions': [dimension_str],
                        'extensions': [],  # 暂时不处理外延
                        'links': [link for link in graph_data['links'] if link['cluster'] == cluster_id]  # 保存连接信息
                    })
    
    # 第二步：去除完全重复的核心聚类
    unique_core_clusters = []
    seen_node_sets = set()
    
    for cluster in core_clusters:
        node_set_key = frozenset(cluster['core_nodes'])
        if node_set_key not in seen_node_sets:
            seen_node_sets.add(node_set_key)
            unique_core_clusters.append(cluster)
        else:
            # 如果节点集合已存在，将维度信息合并到已存在的聚类中，只保留最短的维度标识
            for existing_cluster in unique_core_clusters:
                if frozenset(existing_cluster['core_nodes']) == node_set_key:
                    all_dimensions = existing_cluster['core_dimensions'] + cluster['core_dimensions']
                    shortest_dimension = min(all_dimensions, key=len)
                    existing_cluster['core_dimensions'] = [shortest_dimension]
                    break
    
    # 第三步：处理不同维度之间的重叠聚类
    # 按维度分组
    dimension_groups = defaultdict(list)
    for cluster in unique_core_clusters:
        dimension_groups[cluster['core_dimensions'][0]].append(cluster)
    
    final_core_clusters = []
    processed_clusters = set()
    
    # 遍历每个维度组
    dimensions = sorted(dimension_groups.keys())
    for i, dim1 in enumerate(dimensions):
        for cluster1 in dimension_groups[dim1]:
            if id(cluster1) in processed_clusters:
                continue
                
            cluster1_nodes = set(cluster1['core_nodes'])
            overlapping_found = False
            
            # 与其他维度的聚类比较
            for dim2 in dimensions[i+1:]:
                for cluster2 in dimension_groups[dim2]:
                    if id(cluster2) in processed_clusters:
                        continue
                        
                    cluster2_nodes = set(cluster2['core_nodes'])
                    intersection = cluster1_nodes & cluster2_nodes
                    
                    # 如果有显著重叠（>80%）
                    if len(intersection) > 0:
                        overlap_ratio1 = len(intersection) / len(cluster1_nodes)
                        overlap_ratio2 = len(intersection) / len(cluster2_nodes)
                        
                        if overlap_ratio1 > 0.8 or overlap_ratio2 > 0.8:
                            overlapping_found = True
                            # 创建新的核心聚类（重叠部分）
                            new_core = {
                                'core_nodes': list(intersection),
                                'core_dimensions': [min(cluster1['core_dimensions'][0], cluster2['core_dimensions'][0])],
                                'extensions': [],
                                'links': []  # 将在后面更新
                            }
                            
                            # 创建外延（非重叠部分）
                            ext1_nodes = cluster1_nodes - intersection
                            ext2_nodes = cluster2_nodes - intersection
                            
                            if ext1_nodes:
                                new_core['extensions'].append({
                                    'dimension': f"z_{cluster1['core_dimensions'][0]}",
                                    'nodes': list(ext1_nodes)
                                })
                            
                            if ext2_nodes:
                                new_core['extensions'].append({
                                    'dimension': f"z_{cluster2['core_dimensions'][0]}",
                                    'nodes': list(ext2_nodes)
                                })
                            
                            # 更新连接信息
                            new_core['links'] = [
                                link for link in cluster1['links'] + cluster2['links']
                                if (link['source'] in intersection and link['target'] in intersection)
                            ]
                            
                            final_core_clusters.append(new_core)
                            processed_clusters.add(id(cluster1))
                            processed_clusters.add(id(cluster2))
                            break
                
                if overlapping_found:
                    break
            
            # 如果没有找到重叠，直接添加到最终结果
            if not overlapping_found and id(cluster1) not in processed_clusters:
                final_core_clusters.append(cluster1)
                processed_clusters.add(id(cluster1))
    
    # 第四步：对外延进行去重
    for cluster in final_core_clusters:
        unique_extensions = []
        seen_ext_nodes = set()
        
        for ext in cluster['extensions']:
            ext_nodes = frozenset(ext['nodes'])
            if ext_nodes not in seen_ext_nodes:
                seen_ext_nodes.add(ext_nodes)
                unique_extensions.append(ext)
        
        cluster['extensions'] = unique_extensions
    
    # 生成可视化数据结构
    nodes = []
    links = []
    
    # 添加所有节点
    for i, cluster in enumerate(final_core_clusters):
        core_id = f"core_{i}"
        nodes.append({
            "id": core_id,
            "name": f"核心聚类 {i+1}",
            "type": "core",
            "dimensions": cluster['core_dimensions'],
            "size": len(cluster['core_nodes'])
        })
        
        # 添加外延节点
        for j, ext in enumerate(cluster['extensions']):
            ext_id = f"ext_{i}_{j}"
            nodes.append({
                "id": ext_id,
                "name": f"外延({ext['dimension']})",
                "type": "extension",
                "dimension": ext['dimension'],
                "size": len(ext['nodes'])
            })
            
            # 添加核心到外延的连接
            links.append({
                "source": core_id,
                "target": ext_id,
                "value": 1
            })
    
    # 保存结果
    final_graph_data = {
        'core_clusters': final_core_clusters,
        'total_cores': len(final_core_clusters),
        'visualization': {
            'nodes': nodes,
            'links': links
        }
    }
    
    output_file = os.path.join(subgraphs_dir, 'subgraph_dimension_all.json')
    with open(output_file, 'w') as f:
        json.dump(final_graph_data, f, indent=4)
    
    print(f"核心聚类分析完成，共找到 {len(final_core_clusters)} 个核心聚类")
    return final_graph_data

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
    
    # 分析聚类重叠
    analyze_cluster_overlaps(subgraphs_dir)

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