
import pandas as pd
import numpy as np


def compute_hue_salience(h_normalized, H_ref=0.0, sigma=0.25):
    delta_h = min(abs(h_normalized - H_ref), 1 - abs(h_normalized - H_ref))
    hue_salience = np.exp(- (delta_h / sigma) ** 2)
    return hue_salience

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
    df['tag'] = df['tag'] / 100.0

    # 2. 不透明度归一化
    df['opacity'] = np.sqrt(df['opacity'])

    # 3. 颜色归一化
    # 色相归一化
    df['fill_h_n'] = df['fill_h'] / 360.0
    df['stroke_h_n'] = df['stroke_h'] / 360.0

    # 饱和度归一化
    df['fill_s_n'] = df['fill_s'] / 100.0
    df['stroke_s_n'] = df['stroke_s'] / 100.0

    # 亮度归一化
    df['fill_l_n'] = df['fill_l'] / 100.0
    df['stroke_l_n'] = df['stroke_l'] / 100.0

    # 计算颜色显著性
    w_H, w_S, w_L = 0.4, 0.3, 0.3  # 权重
    H_ref = 0.0  # 红色为参考色相

    sigma = 0.25  # 控制色相显著性曲线宽度的参数

    # 填充色显著性
    df['fill_hue_salience'] = df['fill_h_n'].apply(lambda h: compute_hue_salience(h, H_ref, sigma))
    df['fill_saturation_salience'] = df['fill_s_n']
    df['fill_lightness_salience'] = 1 - abs(df['fill_l_n'] - 0.5) / 0.5
    df['fill_color_sal'] = (w_H * df['fill_hue_salience'] +
                                 w_S * df['fill_saturation_salience'] +
                                 w_L * df['fill_lightness_salience'])

    # 描边色显著性
    df['stroke_hue_salience'] = df['stroke_h_n'].apply(lambda h: compute_hue_salience(h, H_ref, sigma))
    df['stroke_saturation_salience'] = df['stroke_s_n']
    df['stroke_lightness_salience'] = 1 - abs(df['stroke_l_n'] - 0.5) / 0.5
    df['stroke_color_sal'] = (w_H * df['stroke_hue_salience'] +
                                   w_S * df['stroke_saturation_salience'] +
                                   w_L * df['stroke_lightness_salience'])

    # 4. 描边宽度归一化
    max_stroke_width = df['stroke_width'].max() if df['stroke_width'].max() > 0 else 1.0
    df['stroke_width'] = np.sqrt(df['stroke_width'] / max_stroke_width)

    # 5. 图层显著性归一化
    lambda_decay = 0.5  # 衰减系数
    df['layer_list'] = df['layer'].apply(eval)
    max_depth = df['layer_list'].apply(len).max()
    max_indices_per_level = [0] * max_depth
    for i in range(max_depth):
        max_indices_per_level[i] = df['layer_list'].apply(lambda x: x[i] if i < len(x) else 0).max()

    def compute_layer_salience(layer_list):
        salience = 0.0
        for i, idx in enumerate(layer_list):
            max_idx = max_indices_per_level[i]
            if max_idx > 0:
                normalized_idx = 1 - (idx / max_idx)
                salience += normalized_idx * (lambda_decay ** i)
            else:
                salience += 0
        return salience

    df['layer'] = df['layer_list'].apply(compute_layer_salience)

    # 6. 边界框位置和尺寸归一化
    svg_center_x = (svg_min_left + svg_max_right) / 2.0
    svg_center_y = (svg_min_top + svg_max_bottom) / 2.0

    df['bbox_left_n'] = (df['bbox_min_left'] - svg_center_x) / (svg_width / 2.0)
    df['bbox_right_n'] = (df['bbox_max_right'] - svg_center_x) / (svg_width / 2.0)
    df['bbox_top_n'] = (df['bbox_min_top'] - svg_center_y) / (svg_height / 2.0)
    df['bbox_bottom_n'] = (df['bbox_max_bottom'] - svg_center_y) / (svg_height / 2.0)

    df['bbox_center_x_n'] = (df['bbox_center_x'] - svg_center_x) / (svg_width / 2.0)
    df['bbox_center_y_n'] = (df['bbox_center_y'] - svg_center_y) / (svg_height / 2.0)

    df['bbox_width_n'] = df['bbox_width'] / svg_width
    df['bbox_height_n'] = df['bbox_height'] / svg_height

    # 7. 面积归一化
    df['bbox_fill_area'] = np.log1p(df['bbox_fill_area']) / np.log1p(svg_area)
    df['bbox_stroke_area'] = (np.log1p(df['bbox_stroke_area']) / np.log1p(svg_area)) * 0.3

    # 8. 保存归一化后的特征数据
    normalized_columns = [
        'tag_name', 'tag', 'opacity',
        'fill_h_n', 'fill_s_n', 'fill_l_n', 'fill_color_sal',
        'stroke_h_n', 'stroke_s_n', 'stroke_l_n', 'stroke_color_sal', 'stroke_width',
        'layer', 'bbox_left_n', 'bbox_right_n', 'bbox_top_n',
        'bbox_bottom_n', 'bbox_center_x_n', 'bbox_center_y_n',
        'bbox_width_n', 'bbox_height_n', 'bbox_fill_area', 'bbox_stroke_area'
    ]
    df[normalized_columns].to_csv(output_path, index=False)
