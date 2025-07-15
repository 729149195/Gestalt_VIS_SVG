import os
import json
import pandas as pd
from collections import defaultdict

# 定义目录路径
questionnaire_dir = './DataProduce/QuestionnaireData_3'
features_dir = 'Questionnaire_normal_features_train40_mds_211_v2'
step_groups_dir = './DataProduce/StepGroups_mds_211_train40_v2'
output_dir = './DataProduce/UpdatedStepGroups_mds_211_train40_v2'

# 创建输出文件夹
os.makedirs(step_groups_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Step 1: 生成 StepGroups
step_groups = defaultdict(list)

# 遍历问卷数据文件
for filename in os.listdir(questionnaire_dir):
    if filename.endswith(".json"):
        file_path = os.path.join(questionnaire_dir, filename)
        
        # 打开并加载 JSON 文件
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            
            # 对每个step进行处理
            for step in data.get("steps", []):
                step_id = step.get("stepId")
                for group in step.get("groups", []):
                    nodes = group.get("nodes", [])
                    ratings = group.get("ratings", {})
                    correlation_strength = ratings.get("correlation_strength")
                    exclusionary_force = ratings.get("exclusionary_force")
                    # 将节点列表和评分组合
                    nodes_with_ratings = nodes + [correlation_strength, exclusionary_force]
                    step_groups[step_id].append(nodes_with_ratings)

# 将每个 stepId 的组合结果保存为单独的 JSON 文件
for step_id, groups in step_groups.items():
    output_file_path = os.path.join(step_groups_dir, f"step_{step_id}.json")
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        json.dump(groups, output_file, ensure_ascii=False, indent=2)

print("StepGroups 文件夹生成完成。")

# Step 2: 替换节点名称为特征向量，生成 UpdatedStepGroups
for step_file in os.listdir(step_groups_dir):
    if step_file.endswith(".json") and '_' in step_file:
        try:
            step_id = step_file.split('_')[1].split('.')[0]  # 提取 stepId
        except IndexError:
            print(f"Skipping file {step_file} due to naming format issue.")
            continue

        step_file_path = os.path.join(step_groups_dir, step_file)
        
        # 读取 JSON 文件中的节点信息
        with open(step_file_path, 'r', encoding='utf-8') as file:
            step_data = json.load(file)
        
        # 读取对应的 CSV 文件
        csv_file_path = os.path.join(features_dir, f"{step_id}.csv")
        if not os.path.exists(csv_file_path):
            print(f"Warning: CSV file for stepId {step_id} not found.")
            continue
        
        # 加载 CSV 并设置 tag_name 为索引
        df = pd.read_csv(csv_file_path)
        
        # 获取特征列名（除去 'tag_name' 列）
        feature_columns = df.columns.tolist()[1:]
        
        # 创建字典以存储 tag_name 的后缀映射至特征向量（特征顺序与 all_features 一致）
        tag_suffix_to_features = {}
        for index, row in df.iterrows():
            tag_name = row['tag_name']
            tag_suffix = tag_name.split('/')[-1]
            feature_vector = row[feature_columns].tolist()
            tag_suffix_to_features[tag_suffix] = feature_vector

        # 将 CSV 文件中的全部特征作为二维数组复制进入 JSON（特征顺序与 feature_columns 一致）
        all_features = df[feature_columns].values.tolist()

        # 替换 nodes 中的每个节点
        updated_step_data = []
        for group in step_data:
            nodes_with_ratings = group
            nodes = nodes_with_ratings[:-2]  # 分离节点
            ratings = nodes_with_ratings[-2:]  # 分离评分
            
            updated_group = []
            for node in nodes:
                if node in tag_suffix_to_features:
                    feature_vector = tag_suffix_to_features[node]
                    updated_group.append(feature_vector)
            
            updated_group.extend(ratings)  # 添加回评分
            updated_step_data.append(updated_group)
        
        # 构建最终的 JSON 数据
        final_output = {
            "all_features": all_features,
            "groups": updated_step_data
        }
        
        # 将更新后的数据写入新的 JSON 文件
        output_file_path = os.path.join(output_dir, f"step_{step_id}.json")
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            json.dump(final_output, output_file, ensure_ascii=False, indent=2)

print("所有 stepId 文件节点替换完成，文件已保存到 UpdatedStepGroups 文件夹下。")
