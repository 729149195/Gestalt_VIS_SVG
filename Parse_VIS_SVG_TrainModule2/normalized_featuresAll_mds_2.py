import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.manifold import MDS
import json

def simple_mds_embedding(bbox_features):
    """
    简化的MDS嵌入实现，直接使用边界框特征
    """
    mds = MDS(n_components=2, 
              random_state=42,
              metric=True,
              n_init=4)
    
    return mds.fit_transform(bbox_features)

def compute_containment_matrix(df):
    """
    计算包含矩阵：若一个元素的四个边界框都在另一个元素的边界框内就是被包含
    """
    n = len(df)
    containment_matrix = np.zeros((n, n), dtype=np.float32)
    
    # 提取边界框数据为numpy数组，提高计算速度
    bbox_data = df[['bbox_min_left', 'bbox_max_right', 'bbox_min_top', 'bbox_max_bottom']].values
    
    for i in range(n):
        # 向量化计算：检查元素i是否被其他元素包含
        # 条件：bbox_min_top >= other_top AND bbox_max_bottom <= other_bottom 
        #      AND bbox_min_left >= other_left AND bbox_max_right <= other_right
        contained_mask = (
            (bbox_data[i, 2] >= bbox_data[:, 2]) &  # min_top >= other_min_top
            (bbox_data[i, 3] <= bbox_data[:, 3]) &  # max_bottom <= other_max_bottom
            (bbox_data[i, 0] >= bbox_data[:, 0]) &  # min_left >= other_min_left
            (bbox_data[i, 1] <= bbox_data[:, 1])    # max_right <= other_max_right
        )
        containment_matrix[i, contained_mask] = 1.0
        # 自己不能包含自己
        containment_matrix[i, i] = 0.0
    
    return containment_matrix

def compute_adjacency_matrix(df):
    """
    计算邻接矩阵：基于边界框和描边宽度计算元素间的邻接关系
    邻接定义：两个元素在某个方向上接触，且在另一个方向上有重叠
    """
    n = len(df)
    adjacency_matrix = np.zeros((n, n), dtype=np.float32)
    
    # 提取需要的数据
    bbox_data = df[['bbox_min_left', 'bbox_max_right', 'bbox_min_top', 'bbox_max_bottom', 'stroke_width']].values
    
    # 计算每个元素的调整后边界（考虑描边宽度的一半）
    half_stroke = bbox_data[:, 4] / 2.0
    left_adj = bbox_data[:, 0] - half_stroke
    right_adj = bbox_data[:, 1] + half_stroke
    top_adj = bbox_data[:, 2] - half_stroke
    bottom_adj = bbox_data[:, 3] + half_stroke
    
    # 设置容差值，用于浮点数比较
    tolerance = 1e-6
    
    for i in range(n):
        for j in range(i + 1, n):
            # 检查水平方向上的重叠
            horizontal_overlap = (
                max(left_adj[i], left_adj[j]) < min(right_adj[i], right_adj[j])
            )
            
            # 检查垂直方向上的重叠
            vertical_overlap = (
                max(top_adj[i], top_adj[j]) < min(bottom_adj[i], bottom_adj[j])
            )
            
            # 检查水平方向上的接触
            horizontal_touch = (
                abs(right_adj[i] - left_adj[j]) < tolerance or
                abs(left_adj[i] - right_adj[j]) < tolerance
            )
            
            # 检查垂直方向上的接触
            vertical_touch = (
                abs(bottom_adj[i] - top_adj[j]) < tolerance or
                abs(top_adj[i] - bottom_adj[j]) < tolerance
            )
            
            # 判断邻接关系：
            # 1. 水平接触且垂直有重叠
            # 2. 垂直接触且水平有重叠  
            # 3. 水平和垂直都有重叠（完全重叠）
            is_adjacent = (
                (horizontal_touch and vertical_overlap) or
                (vertical_touch and horizontal_overlap) or
                (horizontal_overlap and vertical_overlap)
            )
            
            if is_adjacent:
                adjacency_matrix[i, j] = 1.0
                adjacency_matrix[j, i] = 1.0  # 邻接关系是对称的
    
    return adjacency_matrix

def matrix_mds_embedding(matrix):
    """
    对矩阵使用MDS降维到1维
    """
    if matrix.shape[0] < 2:
        return np.zeros((matrix.shape[0], 1))
    
    # 检查矩阵是否全为0或者所有行都相同
    if np.all(matrix == 0) or np.all(matrix == matrix[0]):
        return np.zeros((matrix.shape[0], 1))
    
    # 检查矩阵的方差，如果方差太小说明数据没有变化
    if np.var(matrix) < 1e-10:
        return np.zeros((matrix.shape[0], 1))
    
    try:
        mds = MDS(n_components=1, 
                  random_state=42,
                  metric=True,
                  n_init=4,
                  max_iter=300,
                  eps=1e-6)
        
        result = mds.fit_transform(matrix)
        
        # 检查结果是否包含NaN或无穷大
        if np.any(np.isnan(result)) or np.any(np.isinf(result)):
            return np.zeros((matrix.shape[0], 1))
            
        return result
        
    except Exception as e:
        # 如果MDS失败，返回零向量
        print(f"MDS failed: {e}")
        return np.zeros((matrix.shape[0], 1))

def normalize_features(input_path, output_path):
    df = pd.read_csv(input_path)

    # 替换 -1 为一个小正数
    df.replace(-1, 0.000001, inplace=True)

    # 计算 svg_min_left, svg_max_right, svg_min_top, svg_max_bottom
    svg_min_left = df['bbox_min_left'].min()
    svg_max_right = df['bbox_max_right'].max()
    svg_min_top = df['bbox_min_top'].min()
    svg_max_bottom = df['bbox_max_bottom'].max()

    # 计算 svg_width, svg_height, svg_area
    svg_width = svg_max_right - svg_min_left
    svg_height = svg_max_bottom - svg_min_top
    
    # 确保不会出现除以0的情况
    if svg_width == 0:
        svg_width = 1.0
    if svg_height == 0:
        svg_height = 1.0
        
    svg_area = svg_width * svg_height

    # 1. 标签类型归一化
    df['tag'] = df['tag'] / 8.0

    # 2. 不透明度归一化
    df['opacity'] = np.sqrt(df['opacity']) 

    df['fill_h_cos'] = (np.cos(2 * np.pi * df['fill_h'] / 360) + 1) 
    df['fill_h_sin'] = (np.sin(2 * np.pi * df['fill_h'] / 360) + 1) 

    df['stroke_h_cos'] = np.where(df['stroke_width'] == 0, 
                                 0, 
                                 (np.cos(2 * np.pi * df['stroke_h'] / 360) + 1) / 2.0)
    df['stroke_h_sin'] = np.where(df['stroke_width'] == 0,
                                 0,
                                 (np.sin(2 * np.pi * df['stroke_h'] / 360) + 1) / 2.0)

    # 饱和度归一化
    df['fill_s_n'] = df['fill_s'] / 100.0
    df['stroke_s_n'] = np.where(df['stroke_width'] == 0,
                               0,
                               df['stroke_s'] / 100.0)

    # 亮度归一化
    df['fill_l_n'] = df['fill_l'] / 100.0
    df['stroke_l_n'] = np.where(df['stroke_width'] == 0,
                               0,
                               df['stroke_l'] / 100.0)

    # 边界框归一化
    df['bbox_left_n'] = (df['bbox_min_left'] - svg_min_left) / svg_width
    df['bbox_right_n'] = (df['bbox_max_right'] - svg_min_left) / svg_width
    df['bbox_top_n'] = (df['bbox_min_top'] - svg_min_top) / svg_height
    df['bbox_bottom_n'] = (df['bbox_max_bottom'] - svg_min_top) / svg_height

    df['bbox_width_n'] = df['bbox_width'] / svg_width
    df['bbox_height_n'] = df['bbox_height'] / svg_height

    # 7. 面积归一化
    df['bbox_fill_area'] = np.log1p(df['bbox_fill_area']) / np.log1p(svg_area)

    # 4. 描边宽度归一化
    max_stroke_width = df['stroke_width'].max() if df['stroke_width'].max() > 0 else 1.0
    df['stroke_width'] = np.sqrt(df['stroke_width'] / max_stroke_width)

    # 使用简化的MDS实现
    bbox_features = df[['bbox_left_n', 'bbox_right_n', 'bbox_top_n', 'bbox_bottom_n']].values
    bbox_mds = simple_mds_embedding(bbox_features)
    
    # 对MDS特征进行归一化
    df['bbox_mds_1'] = (bbox_mds[:, 0] - bbox_mds[:, 0].min()) / (bbox_mds[:, 0].max() - bbox_mds[:, 0].min())
    df['bbox_mds_2'] = (bbox_mds[:, 1] - bbox_mds[:, 1].min()) / (bbox_mds[:, 1].max() - bbox_mds[:, 1].min())

    # 计算包含矩阵和邻接矩阵
    containment_matrix = compute_containment_matrix(df)
    adjacency_matrix = compute_adjacency_matrix(df)
    
    # # 保存矩阵数据到 JSON 文件
    # # 获取输出目录（与normalized_features.csv相同的目录）
    # output_dir = os.path.dirname(output_path)
    
    # # 创建元素ID列表（从tag_name中提取）
    # element_ids = [tag_name.split('/')[-1] for tag_name in df['tag_name'].tolist()]
    
    # # 保存包含矩阵
    # containment_data = {
    #     'element_ids': element_ids,
    #     'matrix': containment_matrix.tolist(),
    #     'description': 'Containment matrix: element i is contained by element j if matrix[i][j] = 1'
    # }
    # containment_path = os.path.join(output_dir, 'containment_matrix.json')
    # with open(containment_path, 'w', encoding='utf-8') as f:
    #     json.dump(containment_data, f, ensure_ascii=False, indent=2)
    
    # # 保存邻接矩阵
    # adjacency_data = {
    #     'element_ids': element_ids,
    #     'matrix': adjacency_matrix.tolist(),
    #     'description': 'Adjacency matrix: elements i and j are adjacent if matrix[i][j] = 1'
    # }
    # adjacency_path = os.path.join(output_dir, 'adjacency_matrix.json')
    # with open(adjacency_path, 'w', encoding='utf-8') as f:
    #     json.dump(adjacency_data, f, ensure_ascii=False, indent=2)
    
    # print(f"包含矩阵已保存到: {containment_path}")
    # print(f"邻接矩阵已保存到: {adjacency_path}")
    
    # 对矩阵使用MDS降维到1维
    containment_mds = matrix_mds_embedding(containment_matrix)
    adjacency_mds = matrix_mds_embedding(adjacency_matrix)
    
    # 归一化MDS结果
    if containment_mds.shape[0] > 1:
        mds_min = containment_mds.min()
        mds_max = containment_mds.max()
        # 检查是否所有值都相同（避免除零错误）
        if abs(mds_max - mds_min) < 1e-10:
            df['containment_mds'] = 0.0
        else:
            containment_mds_norm = (containment_mds - mds_min) / (mds_max - mds_min)
            df['containment_mds'] = containment_mds_norm.flatten()
    else:
        df['containment_mds'] = 0.0
        
    if adjacency_mds.shape[0] > 1:
        mds_min = adjacency_mds.min()
        mds_max = adjacency_mds.max()
        # 检查是否所有值都相同（避免除零错误）
        if abs(mds_max - mds_min) < 1e-10:
            df['adjacency_mds'] = 0.0
        else:
            adjacency_mds_norm = (adjacency_mds - mds_min) / (mds_max - mds_min)
            df['adjacency_mds'] = adjacency_mds_norm.flatten()
    else:
        df['adjacency_mds'] = 0.0

    n_columns = [
        'tag_name', 
        'tag', 'opacity',
        'fill_h_cos', 'fill_h_sin', 'fill_s_n', 'fill_l_n',
        'stroke_h_cos', 'stroke_h_sin', 'stroke_s_n', 'stroke_l_n', 
        'stroke_width',
        'bbox_left_n', 'bbox_right_n', 'bbox_top_n', 'bbox_bottom_n',
        'bbox_mds_1', 'bbox_mds_2',
        'containment_mds', 'adjacency_mds',
        'bbox_width_n', 'bbox_height_n', 'bbox_fill_area'
    ]
    df[n_columns].to_csv(output_path, index=False)

def process_all_features(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    files = [file_name for file_name in os.listdir(input_dir) if file_name.endswith('.csv')]

    for file_name in tqdm(files, desc="Processing files"):
        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, file_name)
        normalize_features(input_path, output_path)

# 示例使用
input_dir = './Questionnaire_features_40'
output_dir = './Questionnaire_normal_features_train40_mds_211_v2'
process_all_features(input_dir, output_dir)
