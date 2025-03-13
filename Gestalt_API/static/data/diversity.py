import sys
import os

# 添加父目录到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from modules import normalized_features_liner_mds_2 as normalized_features
from modules import featureCSV
import shutil
from tqdm import tqdm
import traceback

def process_svg_files():
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 定义源目录和目标目录
    source_dirs = [
        os.path.join(current_dir, 'newData3'),
        os.path.join(current_dir, 'newData5')
    ]
    output_dir = os.path.join(current_dir, '60svg_normal')
    
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 记录处理失败的文件
    failed_files = []
    
    # 处理每个源目录中的SVG文件
    for source_dir in source_dirs:
        # 获取目录名作为前缀
        dir_prefix = os.path.basename(source_dir)
        
        # 获取目录中的所有SVG文件
        svg_files = [f for f in os.listdir(source_dir) if f.endswith('.svg')]
        
        # 处理每个SVG文件
        for svg_file in tqdm(svg_files, desc=f"处理 {os.path.basename(source_dir)} 中的SVG文件"):
            svg_input_path = os.path.join(source_dir, svg_file)
            
            # 创建临时文件路径
            temp_dir = os.path.join(current_dir, 'temp')
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
                
            # 定义临时文件和输出文件的路径
            features_csv_path = os.path.join(temp_dir, f"{svg_file.replace('.svg', '_features.csv')}")
            svg_with_ids_path = os.path.join(temp_dir, f"{svg_file.replace('.svg', '_with_ids.svg')}")
            normalized_features_path = os.path.join(output_dir, f"{dir_prefix}_{svg_file.replace('.svg', '_normalized.csv')}")
            
            try:
                # 提取特征
                featureCSV.process_and_save_features(svg_input_path, features_csv_path, svg_with_ids_path)
                
                # 归一化特征
                normalized_features.normalize_features(features_csv_path, normalized_features_path)
                
                print(f"成功处理 {svg_file}")
            except Exception as e:
                print(f"处理 {svg_file} 时出错: {e}")
                print(traceback.format_exc())
                failed_files.append((svg_file, str(e)))
    
    # 清理临时目录
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    
    # 输出处理结果统计
    total_files = sum(len([f for f in os.listdir(source_dir) if f.endswith('.svg')]) for source_dir in source_dirs)
    processed_files = total_files - len(failed_files)
    
    print("\n处理结果统计:")
    print(f"总文件数: {total_files}")
    print(f"成功处理: {processed_files}")
    print(f"处理失败: {len(failed_files)}")
    
    if failed_files:
        print("\n处理失败的文件:")
        for file, error in failed_files:
            print(f"- {file}: {error}")
    
    print("\n所有SVG文件处理完成！")

if __name__ == "__main__":
    process_svg_files()

