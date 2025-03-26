import requests
import json

url = 'http://127.0.0.1:5000/modify_and_calculate_salience'
headers = {'Content-Type': 'application/json'}
data = {
    "modify_elements": [
        {
            "ids": ["circle_25", "circle_26", "circle_27", "circle_28", "circle_29", "circle_30","circle_31", "circle_32", "circle_33", "circle_34", "circle_35", "circle_36"],
            "attributes": {"stroke-width": "4.1"}
        }
    ],
    "debug": False  # 启用debug标志，请求详细日志
}

response = requests.post(url, headers=headers, json=data)
print(f"Status Code: {response.status_code}")

# 格式化输出JSON响应
if response.status_code == 200:
    resp_data = response.json()
    print(f"\n响应状态: {resp_data['success']}")
    print(f"计算得到的显著性值: {resp_data['salience']}")
    
    if 'debug_info' in resp_data:
        debug_info = resp_data['debug_info']
        print("\n============ 调试信息 ============")
        print(f"SVG文件路径: {debug_info.get('svg_file_path', 'N/A')}")
        print(f"总共解析出的元素数: {debug_info.get('scope_elements_count', 'N/A')}")
        
        print(f"\n要修改的元素ID: {debug_info.get('modify_ids', 'N/A')}")
        
        print(f"\n高亮元素数量: {debug_info.get('highlighted_count', 'N/A')}")
        print(f"非高亮元素数量: {debug_info.get('non_highlighted_count', 'N/A')}")
        
        print(f"\n高亮元素ID: {debug_info.get('highlighted_ids', 'N/A')}")
        print(f"非高亮元素ID: {debug_info.get('non_highlighted_ids', 'N/A')}")
        
        print(f"\n组内相似度: {debug_info.get('intra_group_similarity', 'N/A')}")
        print(f"组间相似度: {debug_info.get('inter_group_similarity', 'N/A')}")
        print(f"基础显著性分数: {debug_info.get('salience_score_base', 'N/A')}")
        
        print(f"\n所有元素平均面积: {debug_info.get('all_elements_avg_area', 'N/A')}")
        print(f"高亮元素平均面积: {debug_info.get('highlighted_avg_area', 'N/A')}")
        print(f"面积阈值: {debug_info.get('area_threshold', 'N/A')}")
        
        print(f"\n应用面积惩罚: {debug_info.get('applied_area_penalty', 'N/A')}")
        if 'salience_score_after_area' in debug_info:
            print(f"面积惩罚后的显著性: {debug_info['salience_score_after_area']}")
        
        # 输出聚类匹配相关信息
        print(f"\n===== API聚类匹配信息 =====")
        print(f"是否添加额外分数: {debug_info.get('bonus_added', 'N/A')}")
        if debug_info.get('cluster_match_found', False):
            print(f"找到匹配的聚类: {debug_info.get('matched_cluster', 'N/A')}")
        
        print(f"\n加成后的显著性: {debug_info.get('salience_score_after_bonus', 'N/A')}")
        print(f"最终归一化显著性: {debug_info.get('final_normalized_score', 'N/A')}")
        
        print(f"\n===== 与目标值比较 =====")
        print(f"目标显著性值: {debug_info.get('target_score', 'N/A')}")
        print(f"差异: {debug_info.get('difference', 'N/A')}")
        print(f"对应原始显著性分数: {debug_info.get('target_salience_raw', 'N/A')}")
        
        # 打印归一化特征数据
        if 'normalized_features' in debug_info:
            print("\n===== 元素归一化特征 =====")
            features = debug_info['normalized_features']
            for item in features:
                element_id = item['id']
                # 提取ID的最后部分进行比较
                id_parts = element_id.split('/')
                item_id_last_part = id_parts[-1]
                
                # 判断是否是高亮元素
                is_highlighted = item_id_last_part in debug_info.get('highlighted_ids', [])
                
                print(f"\n元素ID: {item_id_last_part} {'(高亮)' if is_highlighted else ''}")
                print(f"特征向量: {item['features']}")
                
                # 打印面积特征（索引19）
                if len(item['features']) > 19:
                    print(f"面积特征(索引19): {item['features'][19]}")
else:
    print(f"错误: {response.text}") 