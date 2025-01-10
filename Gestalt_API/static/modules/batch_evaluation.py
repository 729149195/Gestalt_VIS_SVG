import os
import json
import matplotlib.pyplot as plt
from typing import Dict, List
from evaluation import ClusterEvaluator
import numpy as np
from tqdm import tqdm
import sys
import traceback
from pathlib import Path

# 添加父目录到系统路径以导入app
sys.path.append(str(Path(__file__).parent.parent.parent))
from app import process_svg_file

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 使用微软雅黑
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

class BatchEvaluator:
    def __init__(self, svg_dir: str, ground_truth_dir: str, output_dir: str):
        """
        初始化批量评估器
        
        Args:
            svg_dir: SVG文件目录
            ground_truth_dir: 人工标注结果目录
            output_dir: 输出结果目录
        """
        self.svg_dir = svg_dir
        self.ground_truth_dir = ground_truth_dir
        self.output_dir = output_dir
        self.results = {}
        
    def process_single_file(self, svg_filename: str) -> Dict:
        """
        处理单个SVG文件并评估结果
        
        Args:
            svg_filename: SVG文件名
            
        Returns:
            评估结果字典
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
            evaluator = ClusterEvaluator(
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
        
        Returns:
            所有文件的评估结果
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
        
        Args:
            save_path: 保存图表的路径
        """
        if not self.results:
            print("没有可视化的结果")
            return
            
        # 提取数据
        files = list(self.results.keys())
        abms_scores = [result['ABMS'] for result in self.results.values()]
        wcr_scores = [result['WCR'] for result in self.results.values()]
        final_scores = [result['final_score'] for result in self.results.values()]
        
        # 创建图表
        plt.figure(figsize=(15, 8))
        x = np.arange(len(files))
        width = 0.25
        
        plt.bar(x - width, abms_scores, width, label='ABMS', color='skyblue')
        plt.bar(x, wcr_scores, width, label='WCR', color='lightgreen')
        plt.bar(x + width, final_scores, width, label='最终分数', color='salmon')
        
        plt.xlabel('SVG文件')
        plt.ylabel('评分')
        plt.title('批量评估结果')
        plt.xticks(x, [f.split('.')[0] for f in files], rotation=45)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 添加平均分数标注
        avg_text = f'平均分数:\nABMS: {np.mean(abms_scores):.3f}\nWCR: {np.mean(wcr_scores):.3f}\n最终分数: {np.mean(final_scores):.3f}'
        plt.text(0.02, 0.98, avg_text, transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"结果已保存到: {save_path}")
        else:
            plt.show()
            
        # 输出详细的统计信息
        print("\n详细统计信息:")
        print(f"文件数量: {len(files)}")
        print(f"ABMS - 平均: {np.mean(abms_scores):.3f}, 标准差: {np.std(abms_scores):.3f}")
        print(f"WCR - 平均: {np.mean(wcr_scores):.3f}, 标准差: {np.std(wcr_scores):.3f}")
        print(f"最终分数 - 平均: {np.mean(final_scores):.3f}, 标准差: {np.std(final_scores):.3f}")
        
        # 保存详细结果到JSON文件
        if save_path:
            json_path = save_path.rsplit('.', 1)[0] + '_detailed.json'
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'individual_results': self.results,
                    'statistics': {
                        'file_count': len(files),
                        'abms': {
                            'mean': float(np.mean(abms_scores)),
                            'std': float(np.std(abms_scores))
                        },
                        'wcr': {
                            'mean': float(np.mean(wcr_scores)),
                            'std': float(np.std(wcr_scores))
                        },
                        'final_score': {
                            'mean': float(np.mean(final_scores)),
                            'std': float(np.std(final_scores))
                        }
                    }
                }, f, indent=2, ensure_ascii=False)
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
    evaluator = BatchEvaluator(svg_dir, ground_truth_dir, output_dir)
    
    # 运行批量评估
    evaluator.batch_evaluate()
    
    # 可视化结果并保存
    results_path = os.path.join(output_dir, "batch_evaluation_results.png")
    evaluator.visualize_results(results_path)

if __name__ == '__main__':
    main() 