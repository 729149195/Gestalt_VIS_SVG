import os
import json
import numpy as np
import torch

def count_arrays_in_jsons(directory_path, output_file=None):
    """
    统计指定目录下所有JSON文件中的数组总数，包括去重后的数组数量
    去重时，忽略每个数组最后的两个数字
    
    Args:
        directory_path: JSON文件所在目录路径
        output_file: 输出结果的文件路径, 如果为None则只打印到控制台
        
    Returns:
        total_arrays: 所有JSON文件中包含的数组总数
        total_unique_arrays: 去重后的数组总数
    """
    total_arrays = 0
    total_unique_arrays = 0
    file_count = 0
    file_stats = {}
    
    # 检查目录是否存在
    if not os.path.exists(directory_path):
        print(f"目录 {directory_path} 不存在!")
        return total_arrays, total_unique_arrays
    
    print(f"开始统计目录 {directory_path} 中的JSON文件...")
    
    # 遍历目录中的所有文件
    for filename in os.listdir(directory_path):
        if filename.startswith("step_") and filename.endswith(".json"):
            file_path = os.path.join(directory_path, filename)
            file_count += 1
            
            try:
                # 读取JSON文件
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    
                # 统计数组数量
                if isinstance(data, list):
                    array_count = len(data)
                    total_arrays += array_count
                    
                    # 去重处理（忽略每个数组最后的两个数字）
                    unique_arrays = set()
                    for arr in data:
                        if isinstance(arr, list) and len(arr) >= 2:
                            # 转换为元组并忽略最后两个元素
                            arr_tuple = tuple(arr[:-2])
                            unique_arrays.add(arr_tuple)
                        else:
                            # 对于不是列表或长度小于2的情况，直接添加
                            unique_arrays.add(str(arr))
                    
                    unique_count = len(unique_arrays)
                    total_unique_arrays += unique_count
                    
                    file_stats[filename] = {
                        'total': array_count,
                        'unique': unique_count,
                        'duplicates': array_count - unique_count
                    }
                else:
                    file_stats[filename] = {'total': 0, 'unique': 0, 'duplicates': 0}
                    print(f"警告: 文件 {filename} 不是以数组形式存储的数据")
            
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {str(e)}")
    
    # 准备统计信息
    summary = []
    summary.append("\n============= 统计总结 =============")
    summary.append(f"共处理 {file_count} 个文件")
    summary.append(f"总共包含 {total_arrays} 个数组")
    summary.append(f"去重后共有 {total_unique_arrays} 个不同的数组")
    summary.append(f"重复的数组数量: {total_arrays - total_unique_arrays} 个")
    summary.append("====================================\n")
    
    summary.append("各文件中的数组数量:")
    for filename in sorted(file_stats.keys()):
        stats = file_stats[filename]
        summary.append(f"{filename}: 总数 {stats['total']} 个数组, 去重后 {stats['unique']} 个数组, 重复 {stats['duplicates']} 个数组")
    
    # 将结果输出到控制台
    for line in summary:
        print(line)
    
    # 如果指定了输出文件，将结果写入文件
    if output_file:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for line in summary:
                    f.write(line + '\n')
            print(f"\n统计结果已保存到文件: {output_file}")
        except Exception as e:
            print(f"写入文件时出错: {str(e)}")
    
    return total_arrays, total_unique_arrays

def count_sample_pairs(directory_path, output_file=None):
    """
    按照All_new_instance_only_t-SNE_ZT.py的方式统计可以提取的正负样本对数量
    适用于包含元素ID字符串的数组
    
    Args:
        directory_path: JSON文件所在目录路径
        output_file: 输出结果的文件路径, 如果为None则只打印到控制台
    
    Returns:
        total_pos_pairs: 总的正样本对数量
        total_neg_pairs: 总的负样本对数量
    """
    total_pos_pairs = 0
    total_neg_pairs = 0
    file_count = 0
    file_stats = {}
    
    # 检查目录是否存在
    if not os.path.exists(directory_path):
        print(f"目录 {directory_path} 不存在!")
        return total_pos_pairs, total_neg_pairs
    
    print(f"开始分析目录 {directory_path} 中JSON文件的样本对...")
    
    # 遍历目录中的所有文件
    for filename in os.listdir(directory_path):
        if filename.startswith("step_") and filename.endswith(".json"):
            file_path = os.path.join(directory_path, filename)
            file_count += 1
            
            pos_pairs = 0
            neg_pairs = 0
            chosen_set = set()
            
            try:
                # 读取JSON文件
                with open(file_path, 'r', encoding='utf-8') as file:
                    groups = json.load(file)
                
                if not isinstance(groups, list):
                    print(f"警告: 文件 {filename} 不是以数组形式存储的数据，跳过")
                    continue
                
                # 创建元素ID到索引的映射
                element_to_idx = {}
                unique_elements = set()
                
                # 收集所有唯一元素
                for group in groups:
                    if isinstance(group, list) and len(group) >= 3:
                        # 跳过最后两个标识符
                        elements = group[:-2]
                        for elem in elements:
                            unique_elements.add(elem)
                
                # 为每个唯一元素分配索引
                for idx, elem in enumerate(sorted(unique_elements)):
                    element_to_idx[elem] = idx
                
                # 处理每个组（数组）获取样本对
                for group in groups:
                    if isinstance(group, list) and len(group) >= 3:
                        pos_identifier = group[-2]
                        neg_identifier = group[-1]
                        
                        # 提取元素（去掉最后两个标识符）
                        elements = group[:-2]
                        
                        if len(elements) < 2:
                            continue  # 至少需要2个元素才能形成对
                        
                        elements_set = set(elements)
                        
                        # 负样本空间：所有不在当前组内的元素
                        neg_sample_space = [elem for elem in unique_elements if elem not in elements_set]
                        
                        # 处理正样本对（同一组内的元素互相配对）
                        for i, e1 in enumerate(elements):
                            # 正样本对：与同组其他元素配对
                            for j in range(i+1, len(elements)):
                                e2 = elements[j]
                                pair_key = (min(element_to_idx[e1], element_to_idx[e2]), 
                                           max(element_to_idx[e1], element_to_idx[e2]), 1)
                                if pair_key not in chosen_set:
                                    chosen_set.add(pair_key)
                                    pos_pairs += 1
                            
                            # 负样本对：与其他组元素配对（根据identifier确定比例）
                            if neg_sample_space:
                                percentage = get_percentage(neg_identifier)
                                num_samples = max(1, int(len(neg_sample_space) * percentage))
                                neg_pairs += min(num_samples, len(neg_sample_space))
                
                file_stats[filename] = {
                    'pos_pairs': pos_pairs,
                    'neg_pairs': neg_pairs,
                    'total_pairs': pos_pairs + neg_pairs
                }
                
                total_pos_pairs += pos_pairs
                total_neg_pairs += neg_pairs
            
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {str(e)}")
    
    # 准备统计信息
    summary = []
    summary.append("\n============= 样本对统计总结 =============")
    summary.append(f"共处理 {file_count} 个文件")
    summary.append(f"正样本对总数: {total_pos_pairs}")
    summary.append(f"负样本对总数: {total_neg_pairs}")
    summary.append(f"样本对总数: {total_pos_pairs + total_neg_pairs}")
    summary.append("==========================================\n")
    
    summary.append("各文件中的样本对数量:")
    for filename in sorted(file_stats.keys()):
        stats = file_stats[filename]
        summary.append(f"{filename}: 正样本对 {stats['pos_pairs']}, 负样本对 {stats['neg_pairs']}, 总计 {stats['total_pairs']}")
    
    # 将结果输出到控制台
    for line in summary:
        print(line)
    
    # 如果指定了输出文件，将结果写入文件
    if output_file:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for line in summary:
                    f.write(line + '\n')
            print(f"\n样本对统计结果已保存到文件: {output_file}")
        except Exception as e:
            print(f"写入文件时出错: {str(e)}")
    
    return total_pos_pairs, total_neg_pairs

def get_percentage(identifier):
    """
    根据identifier确定采样百分比
    """
    if identifier == 3:
        percentage = 1.0
    elif identifier == 2:
        percentage = 0.75
    elif identifier == 1:
        percentage = 0.5
    else:
        percentage = 1.0
    return percentage

if __name__ == "__main__":
    # 指定目录路径
    directory_path = "Gestalt_API/static/data/StepGroups_6"
    
    # 统计数组数量
    print("===== 数组统计 =====")
    array_stats_file = "Gestalt_API/static/data/array_statistics.txt"
    count_arrays_in_jsons(directory_path, array_stats_file)
    
    # 统计正负样本对数量
    print("\n\n===== 样本对统计 =====")
    pair_stats_file = "Gestalt_API/static/data/pair_statistics.txt"
    count_sample_pairs(directory_path, pair_stats_file)
