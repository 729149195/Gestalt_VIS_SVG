import pandas as pd
import numpy as np


def normalize_features(input_path, output_path):
    # 读取特征数据
    df = pd.read_csv(input_path)

    # 替换 -1 为 0.000001
    df.replace(-1, 0.000001, inplace=True)

    # 定义各特征的归一化逻辑
    def normalize_tag(tag, min_val, max_val):
        if min_val != max_val:
            return (tag - min_val) / (max_val - min_val)
        else:
            return 1.0

    def normalize_opacity(opacity):
        return opacity / 10

    # 归一化 Lab 颜色中的 L 分量 (0-100 -> 0-1)
    def normalize_l(value):
        return value / 100.0

    # 归一化 Lab 颜色中的 a, b 分量 (-128, 127 -> 0-1)
    def normalize_ab(value):
        return (value + 128) / 255.0

    def normalize_stroke_width(stroke_width, min_val, max_val):
        if max_val != min_val:
            return (stroke_width - min_val) / (max_val - min_val)
        else:
            return 1.0

    def normalize_layer(layer):
        # 将 layer 字符串转换为数值列表
        layer_list = eval(layer)
        if not layer_list:
            return 0.0
        # 将每个数值按最大值归一化
        max_value = max(layer_list)
        if max_value == 0:
            return 0.0
        normalized_layer_list = [x / max_value for x in layer_list]
        # 将归一化后的数值列表转换为一个数值
        normalized_layer_value = sum([val * (0.1 ** idx) for idx, val in enumerate(normalized_layer_list)])
        return normalized_layer_value

    def min_max_normalize(series):
        min_val = series.min()
        max_val = series.max()
        if min_val != max_val:
            return (series - min_val) / (max_val - min_val)
        else:
            return 1.0

    def normalize_area(value):
        return np.log1p(value)

    # 归一化每一列
    df['tag'] = df['tag'].apply(lambda x: normalize_tag(x, df['tag'].min(), df['tag'].max()))
    df['opacity'] = df['opacity'].apply(normalize_opacity)

    # 归一化 Lab 颜色分量
    df['fill_l'] = df['fill_l'].apply(normalize_l)
    df['fill_a'] = df['fill_a'].apply(normalize_ab)
    df['fill_b'] = df['fill_b'].apply(normalize_ab)

    df['stroke_l'] = df['stroke_l'].apply(normalize_l)
    df['stroke_a'] = df['stroke_a'].apply(normalize_ab)
    df['stroke_b'] = df['stroke_b'].apply(normalize_ab)

    df['stroke_width'] = df['stroke_width'].apply(
        lambda x: normalize_stroke_width(x, df['stroke_width'].min(), df['stroke_width'].max()))
    df['layer'] = df['layer'].apply(normalize_layer)

    # 使用 Min-Max 归一化处理 bbox 相关特征
    bbox_columns = ['bbox_min_top', 'bbox_max_bottom', 'bbox_min_left', 'bbox_max_right', 'bbox_center_x',
                    'bbox_center_y', 'bbox_width', 'bbox_height']
    df[bbox_columns] = df[bbox_columns].apply(min_max_normalize, axis=0)

    df['bbox_fill_area'] = df['bbox_fill_area'].map(normalize_area)
    df['bbox_stroke_area'] = df['bbox_stroke_area'].map(normalize_area)

    # 归一化后，再次应用 Min-Max 归一化，确保范围在 0 到 1 之间，并处理最小值等于最大值的情况
    def robust_min_max_normalize(series):
        min_val = series.min()
        max_val = series.max()
        if min_val != max_val:
            return (series - min_val) / (max_val - min_val)
        else:
            return pd.Series([1.0] * len(series), index=series.index)

    df[['bbox_fill_area', 'bbox_stroke_area']] = df[['bbox_fill_area', 'bbox_stroke_area']].apply(
        robust_min_max_normalize, axis=0)

    # 对 bbox_stroke_area 应用权重因子，降低其影响力
    stroke_area_weight = 0.3
    df['bbox_stroke_area'] = df['bbox_stroke_area'] * stroke_area_weight

    # 保存归一化后的特征数据
    df.to_csv(output_path, index=False)
