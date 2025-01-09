import json
import numpy as np
from typing import List, Dict, Set, Tuple

class ClusterEvaluator:
    def __init__(self, model_output_path: str, ground_truth_path: str):
        """
        初始化评估器
        
        Args:
            model_output_path: 模型输出的JSON文件路径
            ground_truth_path: 人工标注的JSON文件路径
        """
        self.model_clusters = self._load_model_clusters(model_output_path)
        self.ground_truth_clusters = self._load_ground_truth(ground_truth_path)
        
    def _load_model_clusters(self, file_path: str) -> List[Set[str]]:
        """
        加载模型输出的聚类结果
        
        Args:
            file_path: JSON文件路径
            
        Returns:
            聚类组列表，每个组是一个集合
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
    
    def _load_ground_truth(self, file_path: str) -> List[Set[str]]:
        """
        加载人工标注的聚类结果
        
        Args:
            file_path: JSON文件路径
            
        Returns:
            聚类组列表，每个组是一个集合
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # 移除完全重复的组
        unique_clusters = []
        seen = set()
        for group in data:
            # 提取元素名称（前n-2个元素）
            elements = frozenset(str(x) for x in group[:-2])
            if elements not in seen:
                seen.add(elements)
                unique_clusters.append(set(elements))
                
        return unique_clusters
    
    def jaccard_similarity(self, set1: Set[str], set2: Set[str]) -> float:
        """
        计算两个集合的Jaccard相似度
        
        Args:
            set1: 第一个集合
            set2: 第二个集合
            
        Returns:
            Jaccard相似度值
        """
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0
    
    def calculate_best_matching_score(self) -> float:
        """
        计算平均最佳匹配分数(ABMS)
        
        Returns:
            ABMS分数
        """
        if not self.model_clusters:
            return 0.0
            
        total_score = 0.0
        for model_cluster in self.model_clusters:
            best_sim = max(
                self.jaccard_similarity(model_cluster, gt_cluster)
                for gt_cluster in self.ground_truth_clusters
            )
            total_score += best_sim
            
        return total_score / len(self.model_clusters)
    
    def calculate_weighted_coverage_ratio(self) -> float:
        """
        计算加权覆盖率(WCR)
        
        Returns:
            WCR分数
        """
        if not self.ground_truth_clusters:
            return 0.0
            
        total_weighted_sim = 0.0
        total_weight = 0.0
        
        for gt_cluster in self.ground_truth_clusters:
            weight = len(gt_cluster)
            best_sim = max(
                self.jaccard_similarity(gt_cluster, model_cluster)
                for model_cluster in self.model_clusters
            )
            total_weighted_sim += best_sim * weight
            total_weight += weight
            
        return total_weighted_sim / total_weight if total_weight > 0 else 0.0
    
    def evaluate(self) -> Dict[str, float]:
        """
        计算综合评估指标
        
        Returns:
            包含各项评估指标的字典
        """
        abms = self.calculate_best_matching_score()
        wcr = self.calculate_weighted_coverage_ratio()
        final_score = (abms + wcr) / 2
        
        return {
            'ABMS': abms,
            'WCR': wcr,
            'final_score': final_score,
            'quality_level': self._get_quality_level(final_score)
        }
        
    def _get_quality_level(self, score: float) -> str:
        """
        根据分数确定质量等级
        
        Args:
            score: 评估分数
            
        Returns:
            质量等级描述
        """
        if score >= 0.8:
            return '优秀'
        elif score >= 0.6:
            return '良好'
        elif score >= 0.4:
            return '一般'
        else:
            return '需要改进'

def main():
    """
    主函数，用于测试评估器
    """
    evaluator = ClusterEvaluator(
        'static/data/subgraphs/subgraph_dimension_all.json',
        'static/data/StepGroups_3/step_15.json'
    )
    results = evaluator.evaluate()
    
    print("\n聚类评估结果:")
    print(f"ABMS (平均最佳匹配分数): {results['ABMS']:.4f}")
    print(f"WCR (加权覆盖率): {results['WCR']:.4f}")
    print(f"最终综合分数: {results['final_score']:.4f}")
    print(f"质量等级: {results['quality_level']}")

if __name__ == '__main__':
    main() 