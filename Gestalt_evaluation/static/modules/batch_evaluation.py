import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Set, Tuple
import numpy as np
from tqdm import tqdm
import sys
import traceback
from pathlib import Path
from collections import Counter
from scipy.spatial.distance import cosine
import pandas as pd

# 添加父目录到系统路径以导入app
sys.path.append(str(Path(__file__).parent.parent.parent))
from app import process_svg_file

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class EnhancedClusterEvaluator:
    def __init__(self, model_output_path: str, ground_truth_path: str):
        """
        初始化增强版评估器
        
        Args:
            model_output_path: 模型输出的JSON文件路径
            ground_truth_path: 人工标注的JSON文件路径
        """
        self.MATCH_THRESHOLD = 0.5  # 降低匹配阈值
        self.PARTIAL_MATCH_THRESHOLD = 0.3  # 部分匹配阈值
        self.model_clusters = self._load_model_clusters(model_output_path)
        self.human_annotations = self._load_ground_truth(ground_truth_path)
        self.annotation_frequencies = self._calculate_annotation_frequencies()
        self.TOP_K = 3  # 设置topK匹配数
        self.COHESION_WEIGHT = 0.6  # 内聚度权重
        
    def _load_model_clusters(self, file_path: str) -> List[Set[str]]:
        """
        加载并处理模型输出的聚类结果
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        clusters = []
        # 处理核心聚类组
        for core_cluster in data['core_clusters']:
            # 添加核心节点
            core_nodes = set(node.split('/')[-1] for node in core_cluster['core_nodes'])
            clusters.append(core_nodes)
            
            # 如果有外延，创建包含外延的新组
            if core_cluster.get('extensions'):
                extended_nodes = core_nodes.copy()
                for ext in core_cluster['extensions']:
                    ext_nodes = set(node.split('/')[-1] for node in ext['nodes'])
                    extended_nodes.update(ext_nodes)
                clusters.append(extended_nodes)
                
        return clusters
    
    def _load_ground_truth(self, file_path: str) -> List[Tuple[str]]:
        """
        加载人工标注数据，保留重复组
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # 转换为元组列表，保留重复
        annotations = []
        for group in data:
            # 提取元素名称（前n-2个元素）
            elements = tuple(sorted(str(x) for x in group[:-2]))
            annotations.append(elements)
                
        return annotations
    
    def _calculate_annotation_frequencies(self) -> Dict[Tuple[str], float]:
        """
        计算每个标注组的出现频率
        """
        counter = Counter(self.human_annotations)
        total = len(self.human_annotations)
        return {group: count/total for group, count in counter.items()}
    
    def _calculate_similarity(self, set1: Set[str], set2: Set[str]) -> float:
        """
        计算改进版的相似度，考虑部分匹配
        """
        if not set1 or not set2:
            return 0.0
            
        # 计算基础Jaccard相似度
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        jaccard = intersection / union
        
        # 考虑集合大小的差异
        size_ratio = min(len(set1), len(set2)) / max(len(set1), len(set2))
        
        # 组合相似度
        return (jaccard * 0.7 + size_ratio * 0.3)
    
    def calculate_gpc(self) -> float:
        """
        改进的群体感知一致性计算
        """
        weighted_similarities = []
        total_weight = 0
        
        for annotation, freq in self.annotation_frequencies.items():
            annotation_set = set(annotation)
            # 计算与所有模型聚类的相似度
            similarities = [self._calculate_similarity(annotation_set, cluster)
                          for cluster in self.model_clusters]
            
            # 使用软最大值，考虑多个高相似度匹配
            sorted_sims = sorted(similarities, reverse=True)
            if len(sorted_sims) > 1:
                max_sim = sorted_sims[0] * 0.7 + sorted_sims[1] * 0.3
            else:
                max_sim = sorted_sims[0]
                
            # 频率作为权重，并对高频组给予额外奖励
            weight = freq * (1 + freq)  # 频率越高，权重增加越多
            weighted_similarities.append(max_sim * weight)
            total_weight += weight
            
        return sum(weighted_similarities) / total_weight if total_weight > 0 else 0.0
    
    def calculate_pcr(self) -> float:
        """改进的感知覆盖度计算（反向匹配）"""
        unique_annotations = set(self.human_annotations)
        coverage_scores = []
        
        # 对每个模型聚类寻找最佳匹配
        for cluster in self.model_clusters:
            similarities = []
            for annotation in unique_annotations:
                annotation_set = set(annotation)
                sim = self._calculate_similarity(cluster, annotation_set)
                similarities.append((sim, self.annotation_frequencies[annotation]))
            
            # 取topK匹配并加权
            top_matches = sorted(similarities, reverse=True)[:self.TOP_K]
            weighted_score = sum(sim * (weight ** 0.5) for sim, weight in top_matches)  # 频率平方根加权
            coverage_scores.append(weighted_score)
            
        return sum(coverage_scores) / len(self.model_clusters) if self.model_clusters else 0.0
    
    def calculate_cohesion(self) -> float:
        """计算聚类内聚度"""
        cohesion_scores = []
        for cluster in self.model_clusters:
            if len(cluster) < 2:
                continue
            # 计算所有元素对之间的相似度
            elements = list(cluster)
            pairwise_sim = []
            for i in range(len(elements)):
                for j in range(i+1, len(elements)):
                    sim = self._element_similarity(elements[i], elements[j])
                    pairwise_sim.append(sim)
            cohesion_scores.append(np.mean(pairwise_sim) if pairwise_sim else 0)
        return np.mean(cohesion_scores) if cohesion_scores else 0.0

    def calculate_separation(self) -> float:
        """计算聚类间区分度"""
        separation_scores = []
        all_clusters = self.model_clusters
        for i in range(len(all_clusters)):
            for j in range(i+1, len(all_clusters)):
                # 计算两个聚类之间的区分度
                inter_sim = self._calculate_similarity(all_clusters[i], all_clusters[j])
                separation_scores.append(1 - inter_sim)  # 相似度越低，区分度越高
        return np.mean(separation_scores) if separation_scores else 0.0

    def _element_similarity(self, elem1: str, elem2: str) -> float:
        """元素级相似度计算（示例实现）"""
        # 这里可以替换为实际的元素相似度计算逻辑
        # 暂时使用简单共现频率
        co_occurrence = 0
        total = 0
        for annotation in self.human_annotations:
            if elem1 in annotation and elem2 in annotation:
                co_occurrence += 1
            total += 1
        return co_occurrence / total if total > 0 else 0
    
    def _build_relationship_matrix(self, clusters: List[Set[str]]) -> np.ndarray:
        """
        构建元素关系矩阵
        """
        # 获取所有唯一元素
        all_elements = sorted(set().union(*[set(cluster) if isinstance(cluster, (set, tuple)) 
                                          else set([cluster]) for cluster in clusters]))
        n = len(all_elements)
        element_to_idx = {elem: idx for idx, elem in enumerate(all_elements)}
        
        # 初始化矩阵
        matrix = np.zeros((n, n))
        
        # 填充矩阵
        for cluster in clusters:
            cluster_set = set(cluster) if isinstance(cluster, (set, tuple)) else set([cluster])
            for elem1 in cluster_set:
                for elem2 in cluster_set:
                    if elem1 != elem2:
                        i, j = element_to_idx[elem1], element_to_idx[elem2]
                        matrix[i, j] += 1
                        matrix[j, i] += 1
                        
        # 归一化
        row_sums = matrix.sum(axis=1)
        matrix = np.divide(matrix, row_sums[:, np.newaxis], where=row_sums[:, np.newaxis]!=0)
        
        return matrix
    
    def calculate_ssi(self) -> float:
        """
        改进的结构相似性计算
        """
        human_matrix = self._build_relationship_matrix(self.human_annotations)
        model_matrix = self._build_relationship_matrix(self.model_clusters)
        
        # 如果矩阵大小不同，采用动态规划进行最佳匹配
        if human_matrix.shape != model_matrix.shape:
            min_size = min(human_matrix.shape[0], model_matrix.shape[0])
            max_size = max(human_matrix.shape[0], model_matrix.shape[0])
            
            # 调整矩阵大小，保持主要结构
            if human_matrix.shape[0] > min_size:
                human_matrix = self._reduce_matrix(human_matrix, min_size)
            if model_matrix.shape[0] > min_size:
                model_matrix = self._reduce_matrix(model_matrix, min_size)
        
        # 计算结构相似性
        # 1. 计算余弦相似度
        cosine_sim = 1 - cosine(human_matrix.flatten(), model_matrix.flatten())
        
        # 2. 计算结构特征相似度
        feature_sim = self._calculate_structure_features_similarity(human_matrix, model_matrix)
        
        # 组合不同相似度指标
        return 0.6 * cosine_sim + 0.4 * feature_sim
    
    def _reduce_matrix(self, matrix: np.ndarray, target_size: int) -> np.ndarray:
        """
        智能减少矩阵大小，保持主要结构
        """
        # 使用主成分分析或聚类方法减少矩阵大小
        from sklearn.cluster import KMeans
        
        if matrix.shape[0] <= target_size:
            return matrix
            
        # 使用K-means聚类将矩阵压缩到目标大小
        kmeans = KMeans(n_clusters=target_size, random_state=42)
        clusters = kmeans.fit_predict(matrix)
        
        # 创建新矩阵
        reduced_matrix = np.zeros((target_size, target_size))
        for i in range(target_size):
            for j in range(target_size):
                # 计算类间平均关系
                mask_i = clusters == i
                mask_j = clusters == j
                if np.any(mask_i) and np.any(mask_j):
                    reduced_matrix[i, j] = np.mean(matrix[mask_i][:, mask_j])
        
        return reduced_matrix
    
    def _calculate_structure_features_similarity(self, matrix1: np.ndarray, matrix2: np.ndarray) -> float:
        """
        计算矩阵的结构特征相似度
        """
        # 1. 计算度分布相似度
        degrees1 = matrix1.sum(axis=1)
        degrees2 = matrix2.sum(axis=1)
        degree_sim = 1 - np.abs(np.mean(degrees1) - np.mean(degrees2)) / max(np.mean(degrees1), np.mean(degrees2))
        
        # 2. 计算聚类系数相似度
        clustering1 = np.mean(np.diagonal(np.dot(matrix1, matrix1)))
        clustering2 = np.mean(np.diagonal(np.dot(matrix2, matrix2)))
        clustering_sim = 1 - np.abs(clustering1 - clustering2) / max(clustering1, clustering2)
        
        # 3. 计算中心性相似度
        centrality1 = np.std(degrees1)
        centrality2 = np.std(degrees2)
        centrality_sim = 1 - np.abs(centrality1 - centrality2) / max(centrality1, centrality2)
        
        # 组合所有特征相似度
        return (degree_sim * 0.4 + clustering_sim * 0.3 + centrality_sim * 0.3)
    
    def evaluate(self) -> Dict[str, float]:
        """计算综合评估指标"""
        gpc = self.calculate_gpc()
        pcr = self.calculate_pcr()
        ssi = self.calculate_ssi()
        cohesion = self.calculate_cohesion()
        separation = self.calculate_separation()
        
        # 动态权重调整
        cohesion_weight = self.COHESION_WEIGHT * (1 - cohesion)  # 内聚度越低，权重越高
        final_score = (
            gpc * 0.3 + 
            pcr * 0.25 + 
            ssi * 0.25 +
            cohesion * cohesion_weight +
            separation * (1 - cohesion_weight)
        )
        
        return {
            'GPC': gpc,
            'PCR': pcr,
            'SSI': ssi,
            'Cohesion': cohesion,
            'Separation': separation,
            'final_score': final_score,
            'quality_level': self._get_quality_level(final_score)
        }
    
    def _get_quality_level(self, score: float) -> str:
        """
        根据分数确定质量等级
        """
        if score >= 0.8:
            return '优秀'
        elif score >= 0.6:
            return '良好'
        elif score >= 0.4:
            return '一般'
        else:
            return '需要改进'

class EnhancedBatchEvaluator:
    def __init__(self, svg_dir: str, ground_truth_dir: str, output_dir: str):
        """
        初始化批量评估器
        """
        self.svg_dir = svg_dir
        self.ground_truth_dir = ground_truth_dir
        self.output_dir = output_dir
        self.results = {}
        
    def process_single_file(self, svg_filename: str) -> Dict:
        """
        处理单个SVG文件并评估结果
        """
        try:
            # 获取文件编号
            file_number = svg_filename.split('.')[0]
            
            # 构建文件路径
            svg_path = os.path.join(self.svg_dir, svg_filename)
            ground_truth_path = os.path.join(self.ground_truth_dir, f'step_{file_number}.json')
            
            # 使用app.py中的处理函数
            result = process_svg_file(svg_path)
            if not result['success']:
                print(f"处理文件 {svg_filename} 失败: {result.get('error', '未知错误')}")
                return None
                
            # 创建评估器并评估
            evaluator = EnhancedClusterEvaluator(
                os.path.join(self.output_dir, 'subgraphs/subgraph_dimension_all.json'),
                ground_truth_path
            )
            eval_result = evaluator.evaluate()
            
            return eval_result
            
        except Exception as e:
            print(f"评估文件 {svg_filename} 时出错: {str(e)}")
            print(f"错误堆栈: {traceback.format_exc()}")
            return None
            
    def batch_evaluate(self) -> Dict[str, Dict]:
        """
        批量评估所有SVG文件
        """
        try:
            # 获取所有SVG文件
            svg_files = [f for f in os.listdir(self.svg_dir) if f.endswith('.svg')]
            svg_files.sort(key=lambda x: int(x.split('.')[0]))  # 按数字顺序排序
            
            # 批量处理
            print("开始批量评估...")
            for svg_file in tqdm(svg_files):
                result = self.process_single_file(svg_file)
                if result:
                    self.results[svg_file] = result
                    
            return self.results
            
        except Exception as e:
            print(f"批量评估出错: {str(e)}")
            print(f"错误堆栈: {traceback.format_exc()}")
            return {}
    
    def visualize_results(self, save_path: str = None):
        """
        可视化评估结果
        """
        if not self.results:
            print("没有可视化的结果")
            return
            
        # 创建图表网格
        fig = plt.figure(figsize=(25, 18))
        gs = plt.GridSpec(4, 2, height_ratios=[1, 1, 1, 1], figure=fig)
        
        # 1. 整体指标柱状图
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_overall_metrics(ax1)
        
        # 2. 核心指标趋势
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_metrics_trend(ax2, ['GPC', 'PCR', 'SSI'])
        
        # 3. 质量特征趋势
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_metrics_trend(ax3, ['Cohesion', 'Separation'])
        
        # 4. 质量等级分布
        ax4 = fig.add_subplot(gs[2, 0])
        self._plot_quality_distribution(ax4)
        
        # 5. 得分分布
        ax5 = fig.add_subplot(gs[2, 1])
        self._plot_score_distribution(ax5)
        
        # 6. 指标相关性热力图
        ax6 = fig.add_subplot(gs[3, :])
        self._plot_correlation_heatmap(ax6)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"结果已保存到: {save_path}")
            
            # 保存详细结果到JSON
            json_path = save_path.rsplit('.', 1)[0] + '_detailed.json'
            self._save_detailed_results(json_path)
        else:
            plt.show()
    
    def _plot_overall_metrics(self, ax):
        """绘制整体指标柱状图"""
        metrics = ['GPC', 'PCR', 'SSI', 'Cohesion', 'Separation', 'final_score']
        means = [np.mean([r[m] for r in self.results.values() if m in r]) for m in metrics]
        stds = [np.std([r[m] for r in self.results.values() if m in r]) for m in metrics]
        
        bars = ax.bar(metrics, means, yerr=stds, capsize=5)
        ax.set_title('评估指标总体表现')
        ax.set_ylabel('得分')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # 设置y轴范围
        ax.set_ylim(0, 1.0)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom')
    
    def _plot_metrics_trend(self, ax, metrics):
        """通用指标趋势绘制方法"""
        files = list(self.results.keys())
        x = range(1, len(files) + 1)
        
        for metric in metrics:
            values = [self.results[f].get(metric, 0) for f in files]
            ax.plot(x, values, label=metric, marker='o', markersize=4)
            
        ax.set_title(f'{", ".join(metrics)} 变化趋势')
        ax.legend(loc='lower right')
        ax.grid(True)
    
    def _plot_quality_distribution(self, ax):
        """绘制质量等级分布饼图"""
        quality_counts = Counter(r['quality_level'] for r in self.results.values())
        labels = ['优秀', '良好', '一般', '需要改进']
        sizes = [quality_counts.get(label, 0) for label in labels]
        
        # 设置颜色
        colors = ['#2ecc71', '#3498db', '#f1c40f', '#e74c3c']
        
        # 计算百分比
        total = sum(sizes)
        sizes_percent = [size/total*100 for size in sizes]
        
        # 添加百分比到标签
        labels = [f'{label}\n({size:.1f}%)' for label, size in zip(labels, sizes_percent)]
        
        ax.pie(sizes, labels=labels, colors=colors, autopct='', startangle=90)
        ax.set_title('质量等级分布')
    
    def _plot_score_distribution(self, ax):
        """绘制综合得分分布直方图"""
        scores = [r['final_score'] for r in self.results.values()]
        
        # 使用更多的bins使分布更细致
        bins = np.linspace(0, 1, 11)
        ax.hist(scores, bins=bins, edgecolor='black', alpha=0.7)
        ax.set_title('综合得分分布')
        ax.set_xlabel('得分')
        ax.set_ylabel('频次')
        
        # 设置x轴范围
        ax.set_xlim(0, 1.0)
        
        # 添加网格
        ax.grid(True, linestyle='--', alpha=0.7)
    
    def _plot_correlation_heatmap(self, ax):
        """绘制指标相关性热力图"""
        metrics = ['GPC', 'PCR', 'SSI', 'Cohesion', 'Separation', 'final_score']
        data = [[r[m] for m in metrics] for r in self.results.values()]
        df = pd.DataFrame(data, columns=metrics)
        corr = df.corr()
        
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
        ax.set_title('指标相关性分析')
    
    def _save_detailed_results(self, json_path):
        """保存详细结果到JSON文件"""
        detailed_results = {
            'individual_results': self.results,
            'statistics': {
                'file_count': len(self.results),
                'metrics_summary': {
                    metric: {
                        'mean': float(np.mean([r[metric] for r in self.results.values() if metric in r])),
                        'std': float(np.std([r[metric] for r in self.results.values() if metric in r]))
                    }
                    for metric in ['GPC', 'PCR', 'SSI', 'Cohesion', 'Separation', 'final_score']
                },
                'quality_distribution': dict(Counter(r['quality_level'] 
                                                   for r in self.results.values()))
            }
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)
        print(f"详细结果已保存到: {json_path}")

def main():
    """
    主函数，用于运行批量评估
    """
    # 获取当前文件的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建数据目录的绝对路径
    base_dir = os.path.dirname(os.path.dirname(current_dir))  # 回到 Gestalt_API 目录
    
    # 设置绝对路径
    svg_dir = os.path.join(base_dir, "static", "data", "newData3")
    ground_truth_dir = os.path.join(base_dir, "static", "data", "StepGroups_3")
    output_dir = os.path.join(base_dir, "static", "data")
    
    # 创建评估器
    evaluator = EnhancedBatchEvaluator(svg_dir, ground_truth_dir, output_dir)
    
    # 运行批量评估
    evaluator.batch_evaluate()
    
    # 可视化结果并保存
    results_path = os.path.join(output_dir, "enhanced_batch_evaluation_results.png")
    evaluator.visualize_results(results_path)

if __name__ == '__main__':
    main() 