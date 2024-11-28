import os
import pandas as pd
import numpy as np
from tqdm import tqdm

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
    svg_area = svg_width * svg_height

    # 1. 标签类型归一化
    df['tag'] = df['tag'] / 8.0

    # 2. 不透明度归一化
    df['opacity'] = np.sqrt(df['opacity']) * 0.3

    # 3. 颜色归一化
    # 色相归一化
    # df['fill_h_n'] = df['fill_h'] / 360.0
    # df['stroke_h_n'] = df['stroke_h'] / 360.0

    df['fill_h_cos'] = (np.cos(2 * np.pi * df['fill_h'] / 360) + 1) / 2.0
    df['fill_h_sin'] = (np.sin(2 * np.pi * df['fill_h'] / 360) + 1) / 2.0

    df['stroke_h_cos'] = (np.cos(2 * np.pi * df['stroke_h'] / 360) + 1) / 2.0 * 0.3
    df['stroke_h_sin'] = (np.sin(2 * np.pi * df['stroke_h'] / 360) + 1) / 2.0 * 0.3

    # 饱和度归一化
    df['fill_s_n'] = df['fill_s'] / 100.0
    df['stroke_s_n'] = df['stroke_s'] / 100.0

    # 亮度归一化
    df['fill_l_n'] = df['fill_l'] / 100.0
    df['stroke_l_n'] = df['stroke_l'] / 100.0

    # 6. 边界框位置和尺寸归一化
    svg_center_x = (svg_min_left + svg_max_right) / 2.0
    svg_center_y = (svg_min_top + svg_max_bottom) / 2.0

    df['bbox_left_n'] = df['bbox_min_left'] / svg_width
    df['bbox_right_n'] = df['bbox_max_right'] / svg_width
    df['bbox_top_n'] = df['bbox_min_top'] / svg_height
    df['bbox_bottom_n'] = df['bbox_max_bottom'] / svg_height

    df['bbox_center_x_n'] = df['bbox_center_x'] / svg_width
    df['bbox_center_y_n'] = df['bbox_center_y'] / svg_height

    df['bbox_width_n'] = df['bbox_width'] / svg_width
    df['bbox_height_n'] = df['bbox_height'] / svg_height

    # 7. 面积归一化
    df['bbox_fill_area'] = np.log1p(df['bbox_fill_area']) / np.log1p(svg_area)
    df['bbox_stroke_area'] = (np.log1p(df['bbox_stroke_area']) / np.log1p(svg_area)) * 0.3  # w_stroke = 0.3

    # 4. 描边宽度归一化
    max_stroke_width = df['stroke_width'].max() if df['stroke_width'].max() > 0 else 1.0
    df['stroke_width'] = np.sqrt(df['stroke_width'] / 10)

    # 5. 图层显著性归一化
    lambda_decay = 0.5  # 衰减系数
    df['layer'] = df['layer'].apply(eval)
    max_depth = df['layer'].apply(len).max()
    max_indices_per_level = [0] * max_depth
    for i in range(max_depth):
        max_indices_per_level[i] = df['layer'].apply(lambda x: x[i] if i < len(x) else 0).max()

    def compute_layer_sal(layer):
        sal = 0.0
        for i, idx in enumerate(layer):
            max_idx = max_indices_per_level[i]
            if max_idx > 0:
                n_idx = 1 - (idx / max_idx)
                sal += n_idx * (lambda_decay ** i)
            else:
                sal += 0
        return sal

    df['layer_sal'] = df['layer'].apply(compute_layer_sal)

    n_columns = [
        'tag_name', 'tag', 'opacity',
        'fill_h_cos', 'fill_h_sin', 'fill_s_n', 'fill_l_n',
        'stroke_h_cos', 'stroke_h_sin', 'stroke_s_n', 'stroke_l_n', 'stroke_width',
        'layer_sal', 'bbox_left_n', 'bbox_right_n', 'bbox_top_n',
        'bbox_bottom_n', 'bbox_center_x_n', 'bbox_center_y_n',
        'bbox_width_n', 'bbox_height_n', 'bbox_fill_area'
    ]
    df[n_columns].to_csv(output_path, index=False)
