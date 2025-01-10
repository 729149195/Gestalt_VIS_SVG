import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.manifold import MDS

def topology_preserving_distance(X, alignment_threshold=0.05, local_scale=0.1):
    """
    基于多个理论的改进距离计算方法
    参考文献：
    - Dwyer et al. 2009: Constrained Graph Layout by Multidimensional Scaling
    - Kong et al. 2010: Perceptual Guidelines for Creating Rectangular Treemaps
    - van der Maaten & Hinton 2008: t-SNE
    """
    distances = np.zeros((X.shape[0], X.shape[0]))
    
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            # 局部结构权重
            local_weight = np.exp(-np.sum((X[i] - X[j])**2) / (2 * local_scale**2))
            
            # 对齐约束检测
            alignment_constraints = 0
            if abs(X[i,0] - X[j,0]) < alignment_threshold:  # 左对齐
                alignment_constraints += 1
            if abs(X[i,1] - X[j,1]) < alignment_threshold:  # 右对齐
                alignment_constraints += 1
            if abs(X[i,2] - X[j,2]) < alignment_threshold:  # 上对齐
                alignment_constraints += 1
            if abs(X[i,3] - X[j,3]) < alignment_threshold:  # 下对齐
                alignment_constraints += 1
            
            # 综合距离计算
            euclidean_dist = np.sqrt(np.sum((X[i] - X[j])**2))
            topology_factor = 1.0 / (1.0 + alignment_constraints)
            
            distances[i,j] = euclidean_dist * topology_factor * local_weight
    
    return distances

def improved_mds_embedding(bbox_features):
    """
    改进的MDS嵌入实现
    """
    # 计算拓扑保持距离矩阵
    dist_matrix = topology_preserving_distance(bbox_features)
    
    # 使用约束MDS
    mds = MDS(n_components=2, 
              dissimilarity='precomputed',
              random_state=42,
              metric=True,
              n_init=4)
    
    return mds.fit_transform(dist_matrix)

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


    df['fill_h_cos'] = (np.cos(2 * np.pi * df['fill_h'] / 360) + 1) / 2.0
    df['fill_h_sin'] = (np.sin(2 * np.pi * df['fill_h'] / 360) + 1) / 2.0

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

    # 边界框归一化 - 修正计算方式
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

    # 替换原有的MDS实现
    bbox_features = df[['bbox_left_n', 'bbox_right_n', 'bbox_top_n', 'bbox_bottom_n']].values
    bbox_mds = improved_mds_embedding(bbox_features)
    
    # 对MDS特征进行归一化
    df['bbox_mds_1'] = (bbox_mds[:, 0] - bbox_mds[:, 0].min()) / (bbox_mds[:, 0].max() - bbox_mds[:, 0].min())
    df['bbox_mds_2'] = (bbox_mds[:, 1] - bbox_mds[:, 1].min()) / (bbox_mds[:, 1].max() - bbox_mds[:, 1].min())

    n_columns = [
        'tag_name', 
        'tag', 'opacity',
        'fill_h_cos', 'fill_h_sin', 'fill_s_n', 'fill_l_n',
        'stroke_h_cos', 'stroke_h_sin', 'stroke_s_n', 'stroke_l_n', 
        'stroke_width',
        'bbox_left_n', 'bbox_right_n', 'bbox_top_n', 'bbox_bottom_n',
        'bbox_mds_1', 'bbox_mds_2' ,
        'bbox_width_n', 'bbox_height_n', 'bbox_fill_area'
    ]
    df[n_columns].to_csv(output_path, index=False)