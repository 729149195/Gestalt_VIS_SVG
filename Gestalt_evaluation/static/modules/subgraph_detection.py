import json
import os
import numpy as np
import networkx as nx
import community.community_louvain as community_louvain
from collections import Counter, defaultdict
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

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

def calculate_gmm(features):
    """
    通用的GMM计算函数
    
    Args:
        features: 特征数据，shape为(n_samples, n_features)
    
    Returns:
        best_gmm: 最佳GMM模型
        bic_scores: BIC评分列表
        is_single_cluster: 是否所有数据点相同
        scaled_features: 标准化后的特征数据
    """
    # 标准化特征
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # 检查数据是否都相同
    if np.all(scaled_features == scaled_features[0]):
        print("所有数据点相同，归为一个聚类")
        return None, [], True, scaled_features
    
    # 根据数据量动态调整最大聚类数
    max_components = min(5, len(features) - 1)
    if max_components < 2:
        max_components = 2
    
    n_components_range = range(2, max_components + 1)
    bic_scores = []
    gmm_models = []
    
    # 尝试不同的聚类数
    for n_components in n_components_range:
        # 使用更稳定的参数配置
        gmm = GaussianMixture(
            n_components=n_components,
            random_state=42,
            max_iter=200,
            tol=1e-4,
            reg_covar=1e-6  # 增加协方差矩阵的正则化
        )
        gmm.fit(scaled_features)
        bic_scores.append(gmm.bic(scaled_features))
        gmm_models.append(gmm)
    
    # 选择BIC最小的模型
    best_idx = np.argmin(bic_scores)
    best_gmm = gmm_models[best_idx]
    
    return best_gmm, bic_scores, False, scaled_features

def calculate_metrics(features):
    """计算数据的分散度和聚集性
    
    Args:
        features: 特征数据，shape为(n_samples, n_features)
    
    Returns:
        dispersion: 分散度 (0-1)
        clustering: 聚集性 (0-1)
    """
    # 计算分散度 (使用四分位距/数据范围)
    sorted_features = np.sort(features, axis=0)
    q1 = np.percentile(sorted_features, 25, axis=0)
    q3 = np.percentile(sorted_features, 75, axis=0)
    min_vals = np.min(sorted_features, axis=0)
    max_vals = np.max(sorted_features, axis=0)
    
    # 避免除以零
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1  # 防止除以零
    
    # 计算每个维度的分散度
    dimension_dispersions = (q3 - q1) / ranges
    dispersion = np.mean(dimension_dispersions)
    
    # 使用通用GMM函数计算局部聚集性
    best_gmm, _, is_single_cluster, _ = calculate_gmm(features)
    
    # 如果所有数据点相同，聚集性为1
    if is_single_cluster:
        return dispersion, 1.0
    
    # 计算局部聚集性 (使用GMM组件的方差)
    variances = best_gmm.covariances_.reshape(-1)  # 展平协方差矩阵
    weights = 1 / (variances + 1e-6)  # 避免除以零
    total_weight = np.sum(weights)
    normalized_weights = weights / total_weight
    
    clustering = np.sum(normalized_weights * (1 - np.sqrt(variances)))
    
    return dispersion, clustering

def evaluate_dimension_combination(features):
    """评估维度组合的适用性
    
    Args:
        features: 特征数据
    
    Returns:
        result: 评估结果 ("保留" 或 "过滤的原因")
    """
    dispersion, clustering = calculate_metrics(features)
    
    # 根据评估矩阵进行判断
    if clustering > 0.7:  # 高聚集
        if dispersion > 0.7:
            return "保留(多类别特征)", (dispersion, clustering)
        elif dispersion > 0.3:
            return "保留(多类别特征)", (dispersion, clustering)
        else:
            return "需要检查(区分不明显)", (dispersion, clustering)
    elif clustering > 0.3:  # 中等聚集
        if dispersion > 0.7:
            return "可能保留(需要检查)", (dispersion, clustering)
        elif dispersion > 0.3:
            return "需要检查(需要进一步分析)", (dispersion, clustering)
        else:
            return "可能过滤(区分不明显)", (dispersion, clustering)
    else:  # 低聚集
        if dispersion > 0.7:
            return "过滤(纯噪声)", (dispersion, clustering)
        elif dispersion > 0.3:
            return "过滤(杂乱无规律)", (dispersion, clustering)
        else:
            return "过滤(无区分度)", (dispersion, clustering)

def generate_subgraph(identifiers, features, dimensions, clustering_method):
    """为指定维度生成子图"""
    print(f"\n=== 开始生成子图 ===")
    print(f"使用维度: {dimensions}")
    print(f"聚类方法: {clustering_method}")
    
    selected_features = features[:, dimensions]
    
    # 对于GMM方法，提前计算GMM，用于评估和聚类
    if clustering_method.lower() == 'gmm':
        # 计算GMM，仅进行一次
        best_gmm, bic_scores, is_single_cluster, scaled_features = calculate_gmm(selected_features)
        
        # 计算分散度 (与calculate_metrics函数类似)
        sorted_features = np.sort(selected_features, axis=0)
        q1 = np.percentile(sorted_features, 25, axis=0)
        q3 = np.percentile(sorted_features, 75, axis=0)
        min_vals = np.min(sorted_features, axis=0)
        max_vals = np.max(sorted_features, axis=0)
        
        # 避免除以零
        ranges = max_vals - min_vals
        ranges[ranges == 0] = 1  # 防止除以零
        
        # 计算每个维度的分散度
        dimension_dispersions = (q3 - q1) / ranges
        dispersion = np.mean(dimension_dispersions)
        
        # 计算聚集性
        if is_single_cluster:
            clustering = 1.0
        else:
            variances = best_gmm.covariances_.reshape(-1)  # 展平协方差矩阵
            weights = 1 / (variances + 1e-6)  # 避免除以零
            total_weight = np.sum(weights)
            normalized_weights = weights / total_weight
            
            clustering = np.sum(normalized_weights * (1 - np.sqrt(variances)))
    else:
        # 对于非GMM方法，使用原有的评估方式
        evaluation_result, metrics = evaluate_dimension_combination(selected_features)
        dispersion, clustering = metrics
    
    # 使用计算出的指标评估维度组合
    if clustering > 0.7:  # 高聚集
        if dispersion > 0.7:
            evaluation_result = "保留(多类别特征)"
        elif dispersion > 0.3:
            evaluation_result = "保留(多类别特征)"
        else:
            evaluation_result = "需要检查(区分不明显)"
    elif clustering > 0.3:  # 中等聚集
        if dispersion > 0.7:
            evaluation_result = "可能保留(需要检查)"
        elif dispersion > 0.3:
            evaluation_result = "需要检查(需要进一步分析)"
        else:
            evaluation_result = "可能过滤(区分不明显)"
    else:  # 低聚集
        if dispersion > 0.7:
            evaluation_result = "过滤(纯噪声)"
        elif dispersion > 0.3:
            evaluation_result = "过滤(杂乱无规律)"
        else:
            evaluation_result = "过滤(无区分度)"
    
    print(f"维度评估结果: {evaluation_result}")
    print(f"分散度: {dispersion:.3f}, 聚集性: {clustering:.3f}")
    
    # 如果评估结果包含"过滤"，则返回空的图数据结构而不是None
    if "过滤" in evaluation_result:
        print("该维度组合被过滤，跳过聚类")
        return {
            "nodes": [],
            "links": [],
            "clusters": 0,
            "dimensions": dimensions,
            "filtered": True,
            "filter_reason": evaluation_result
        }
    
    communities = {}
    
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
    
    # GMM方法 - 使用之前计算的GMM模型
    elif clustering_method.lower() == 'gmm':
        # 如果所有数据点相同
        if is_single_cluster:
            communities = {identifiers[i]: 0 for i in range(len(identifiers))}
        else:
            # 已经有了之前计算的结果，直接使用
            cluster_labels = best_gmm.predict(scaled_features)
            # 转换为与Louvain方法相同的格式
            communities = {identifiers[i]: int(cluster_labels[i]) for i in range(len(identifiers))}
    
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
        "dimensions": dimensions,
        "filtered": False,
        "filter_reason": None
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
                
                # 跳过被过滤的维度组合
                if graph_data.get('filtered', False):
                    print(f"维度组合 {dimension} 被过滤: {graph_data.get('filter_reason')}")
                    continue
                
                # 确保图数据包含必要的字段
                if not graph_data.get('nodes') or not graph_data.get('links'):
                    print(f"维度组合 {dimension} 的图数据结构不完整，跳过")
                    continue
                
                # 收集该维度下的所有聚类
                clusters = defaultdict(set)
                for node in graph_data['nodes']:
                    clusters[node['cluster']].add(node['id'])
                
                # 将每个聚类作为一个核心聚类
                for cluster_id, nodes in clusters.items():
                    # 转换维度标识
                    dimension_str = str(int(dimension) + 1) if len(dimension) == 1 else ''.join(str(int(d) + 1) for d in dimension)
                    core_clusters.append({
                        'core_nodes': list(nodes),
                        'core_dimensions': [dimension_str],
                        'extensions': [],
                        'links': [link for link in graph_data['links'] if link['cluster'] == cluster_id]
                    })
    
    # 如果没有有效的核心聚类，返回空结果
    if not core_clusters:
        print("没有找到有效的核心聚类")
        return {
            'core_clusters': [],
            'overlap_matrix': [],
            'dimension_groups': {}
        }
    
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
                        
                        if overlap_ratio1 > 0.8 and overlap_ratio2 > 0.8:
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
    
    print(f"The core clustering analysis was completed, finding a total of {len(final_core_clusters)} Core clusters")
    return final_graph_data

def main(features_json_path, output_dir, clustering_method, subgraph_dimensions, progress_callback=None):
    """主函数"""
    print(f"Start processing subgraph detection...")
    print(f"Using clustering methods: {clustering_method}")
    print(f"Dimension combinations: {subgraph_dimensions}")

    if progress_callback:
        progress_callback(85, "Start subgraph detection...")

    # 创建输出目录
    subgraphs_dir = os.path.join(output_dir, 'subgraphs')
    if not os.path.exists(subgraphs_dir):
        os.makedirs(subgraphs_dir)

    # 加载特征数据
    identifiers, features = load_features_from_json(features_json_path)

    total_dimensions = len(subgraph_dimensions)
    # 处理每个维度组合
    for idx, dimensions in enumerate(subgraph_dimensions):
        if progress_callback:
            current_progress = 85 + (idx / total_dimensions) * 10
            progress_callback(current_progress, f"Dimension combinations being processed: {dimensions}")
            
        print(f"\nHandling of dimension combinations: {dimensions}")
        # 生成子图数据
        graph_data = generate_subgraph(identifiers, features, dimensions, clustering_method)
        
        # 保存子图数据
        dimension_str = ''.join(map(str, dimensions))
        subgraph_file = os.path.join(subgraphs_dir, f'subgraph_dimension_{dimension_str}.json')
        with open(subgraph_file, 'w') as f:
            json.dump(graph_data, f, indent=4)
    
    # 分析聚类重叠
    if progress_callback:
        progress_callback(95, "Cluster overlap being analysed...")
    analyze_cluster_overlaps(subgraphs_dir)

    if progress_callback:
        progress_callback(98, "Subgraph detection complete")

if __name__ == '__main__':
    # 配置参数
    features_json_path = '../data/cluster_features.json'
    output_dir = '../data'
    clustering_config = {
        'method': ['louvain', 'gmm'],  # 支持两种聚类方法
        'dimensions': [
            [0], [1], [2], [3],
            [0,1], [0,2], [0,3],
            [1,2], [1,3], [2,3],
            [0,1,2], [0,1,3], [0,2,3], [1,2,3],
            [0,1,2,3]
        ]
    }
    
    # 对每种聚类方法运行主函数
    for method in clustering_config['method']:
        print(f"\n使用聚类方法: {method}")
        output_subdir = os.path.join(output_dir, f'results_{method}')
        if not os.path.exists(output_subdir):
            os.makedirs(output_subdir)
            
        main(
            features_json_path=features_json_path,
            output_dir=output_subdir,
            clustering_method=method,
            subgraph_dimensions=clustering_config['dimensions']
        ) 