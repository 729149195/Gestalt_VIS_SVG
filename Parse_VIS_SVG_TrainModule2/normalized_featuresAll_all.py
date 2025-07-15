import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.manifold import MDS
from colormath.color_objects import LabColor, sRGBColor, HSLColor, LuvColor
from colormath.color_conversions import convert_color

def hsl_to_luv(h, s, l):
    try:
        # HSL -> RGB -> LUV
        hsl = HSLColor(h, s/100, l/100)  # 转换到0-1范围
        rgb = convert_color(hsl, sRGBColor)
        luv = convert_color(rgb, LuvColor)
        
        # 归一化到0-1范围
        # L: 0-100
        # u: -134 到 +224
        # v: -140 到 +122
        l_norm = luv.luv_l / 100.0
        u_norm = (luv.luv_u + 134) / (224 + 134)  # 移动到正区间并归一化
        v_norm = (luv.luv_v + 140) / (122 + 140)  # 移动到正区间并归一化
        
        return l_norm, u_norm, v_norm
    except:
        # 处理异常情况
        return 0, 0.5, 0.5

def compute_visual_attention(df):
    position_saliency = 1 - np.sqrt((df['bbox_center_x_n'] - 0.5)**2 + 
                                   (df['bbox_center_y_n'] - 0.5)**2)
    size_saliency = df['bbox_fill_area']
    color_contrast = np.sqrt(
        (df['fill_luv_l'])**2 + 
        (df['fill_luv_u'] - 0.5)**2 + 
        (df['fill_luv_v'] - 0.5)**2
    )
    
    # 调整权重分配
    attention_score = (0.45 * position_saliency + 
                      0.35 * size_saliency + 
                      0.20 * color_contrast)
    return attention_score

def extract_spatial_features_mds(df):
    spatial_features = df[[
        'bbox_left_n', 'bbox_right_n', 'bbox_top_n', 'bbox_bottom_n',
        'bbox_center_x_n', 'bbox_center_y_n'
    ]].values
    
    n_samples = spatial_features.shape[0]
    distances = np.zeros((n_samples, n_samples))
    
    for i in range(n_samples):
        for j in range(n_samples):
            distances[i,j] = np.sqrt(np.sum((spatial_features[i] - spatial_features[j])**2))
    
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    spatial_2d = mds.fit_transform(distances)
    
    df['spatial_mds_1'] = (spatial_2d[:, 0] - spatial_2d[:, 0].min()) / \
                         (spatial_2d[:, 0].max() - spatial_2d[:, 0].min())
    df['spatial_mds_2'] = (spatial_2d[:, 1] - spatial_2d[:, 1].min()) / \
                         (spatial_2d[:, 1].max() - spatial_2d[:, 1].min())
    return df

def normalize_features(input_path, output_path):
    df = pd.read_csv(input_path)
    df.replace(-1, 0.000001, inplace=True)

    # 基础SVG尺寸计算
    svg_min_left = df['bbox_min_left'].min()
    svg_max_right = df['bbox_max_right'].max()
    svg_min_top = df['bbox_min_top'].min()
    svg_max_bottom = df['bbox_max_bottom'].max()
    svg_width = svg_max_right - svg_min_left
    svg_height = svg_max_bottom - svg_min_top
    svg_area = svg_width * svg_height

    # 标签和不透明度
    df['tag'] = df['tag'] / 8.0
    df['opacity'] = np.sqrt(df['opacity'] * 0.5)

    # 颜色转换和归一化
    # 填充色
    df['fill_h_cos'] = (np.cos(2 * np.pi * df['fill_h'] / 360) + 1) / 2.0
    df['fill_h_sin'] = (np.sin(2 * np.pi * df['fill_h'] / 360) + 1) / 2.0
    df['fill_s_n'] = df['fill_s'] / 100.0
    df['fill_l_n'] = df['fill_l'] / 100.0
    
    # 转换为CIELUV颜色空间
    fill_luv = df.apply(lambda row: hsl_to_luv(row['fill_h'], row['fill_s'], row['fill_l']), axis=1)
    df['fill_luv_l'] = fill_luv.apply(lambda x: x[0])
    df['fill_luv_u'] = fill_luv.apply(lambda x: x[1])
    df['fill_luv_v'] = fill_luv.apply(lambda x: x[2])

    # 描边色
    df['stroke_h_cos'] = np.where(df['stroke_width'] == 0, 
                                 0, 
                                 (np.cos(2 * np.pi * df['stroke_h'] / 360) + 1) / 2.0 * 0.3)
    df['stroke_h_sin'] = np.where(df['stroke_width'] == 0,
                                 0,
                                 (np.sin(2 * np.pi * df['stroke_h'] / 360) + 1) / 2.0 * 0.3)
    df['stroke_s_n'] = np.where(df['stroke_width'] == 0, 0, df['stroke_s'] / 100.0)
    df['stroke_l_n'] = np.where(df['stroke_width'] == 0, 0, df['stroke_l'] / 100.0)

    # 描边色转换为CIELUV（仅当有描边时）
    stroke_luv = df.apply(lambda row: 
        hsl_to_luv(row['stroke_h'], row['stroke_s'], row['stroke_l']) 
        if row['stroke_width'] > 0 else (0, 0.5, 0.5), axis=1)
    df['stroke_luv_l'] = stroke_luv.apply(lambda x: x[0]) * 0.3  # 应用描边权重
    df['stroke_luv_u'] = stroke_luv.apply(lambda x: x[1]) * 0.3
    df['stroke_luv_v'] = stroke_luv.apply(lambda x: x[2]) * 0.3

    # 空间特征归一化
    # 边界框归一化 - 修正计算方式
    df['bbox_left_n'] = (df['bbox_min_left'] - svg_min_left) / svg_width
    df['bbox_right_n'] = (df['bbox_max_right'] - svg_min_left) / svg_width
    df['bbox_top_n'] = (df['bbox_min_top'] - svg_min_top) / svg_height
    df['bbox_bottom_n'] = (df['bbox_max_bottom'] - svg_min_top) / svg_height

    # 中心点坐标归一化
    df['bbox_center_x_n'] = (df['bbox_center_x'] - svg_min_left) / svg_width
    df['bbox_center_y_n'] = (df['bbox_center_y'] - svg_min_top) / svg_height

    df['bbox_width_n'] = df['bbox_width'] / svg_width
    df['bbox_height_n'] = df['bbox_height'] / svg_height

    # 面积特征
    df['bbox_fill_area'] = np.log1p(df['bbox_fill_area']) / np.log1p(svg_area)
    df['bbox_stroke_area'] = (np.log1p(df['bbox_stroke_area']) / np.log1p(svg_area)) * 0.3
    
    # 描边宽度归一化
    max_stroke_width = df['stroke_width'].max() if df['stroke_width'].max() > 0 else 1.0
    df['stroke_width'] = np.sqrt(df['stroke_width'] / max_stroke_width) * 0.3

    # 添加MDS特征
    df = extract_spatial_features_mds(df)
    
    # 添加视觉注意力分数
    df['visual_attention'] = compute_visual_attention(df)

    # 完整的特征列表
    n_columns = [
        # 基础属性
        'tag_name', 'tag', 'opacity',
        
        # 填充色特征 (HSL)
        'fill_h_cos', 'fill_h_sin', 'fill_s_n', 'fill_l_n',
        
        # 填充色特征 (CIELUV)
        'fill_luv_l', 'fill_luv_u', 'fill_luv_v',
        
        # 描边特征 (HSL)
        'stroke_h_cos', 'stroke_h_sin', 'stroke_s_n', 'stroke_l_n',
        
        # 描边特征 (CIELUV)
        'stroke_luv_l', 'stroke_luv_u', 'stroke_luv_v',
        'stroke_width',
        
        # 位置特征
        'bbox_left_n', 'bbox_right_n', 'bbox_top_n', 'bbox_bottom_n',
        'bbox_center_x_n', 'bbox_center_y_n',
        
        # 尺寸特征
        'bbox_width_n', 'bbox_height_n',
        
        # 面积特征
        'bbox_fill_area', 'bbox_stroke_area',
        
        # MDS降维特征
        'spatial_mds_1', 'spatial_mds_2',
        
        # # 视觉注意力特征
        # 'visual_attention'
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
input_dir = './Questionnaire_features_2'
output_dir = './Questionnaire_normal_features_linerposition_all'
process_all_features(input_dir, output_dir)
