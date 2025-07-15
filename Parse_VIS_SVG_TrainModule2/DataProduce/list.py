import os
import json
from pathlib import Path

def count_groups_in_steps():
    # 获取当前工作目录
    current_dir = Path.cwd()
    # 指定目录路径
    directory = current_dir / "DataProduce" / "UpdatedStepGroups_3"
    
    print(f"正在搜索目录: {directory}")
    
    # 存储结果的字典
    results = {}
    
    # 检查目录是否存在
    if not directory.exists():
        print(f"错误：目录 {directory} 不存在")
        return
    
    # 遍历目录中的所有json文件
    files = list(directory.glob("step_*.json"))
    print(f"找到 {len(files)} 个JSON文件")
    
    for file_path in files:
        try:
            # 读取JSON文件
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 获取step编号
            step_num = int(file_path.stem.split('_')[1])
            
            # 统计groups数量
            groups_count = len(data.get('groups', []))
            
            # 存储结果
            results[step_num] = groups_count
            
        except Exception as e:
            print(f"处理文件 {file_path.name} 时出错: {str(e)}")
    
    # 按step编号排序并输出结果
    if not results:
        print("没有找到任何有效的分组数据")
    else:
        for step_num in sorted(results.keys()):
            print(f"Step {step_num}: {results[step_num]} 个分组")

if __name__ == "__main__":
    count_groups_in_steps()
