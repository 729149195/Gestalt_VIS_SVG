import pandas as pd
import numpy as np

def normalize_features(input_path, output_path, output_path_LR, output_path_TB):
    # 读取特征数据
    df = pd.read_csv(input_path)

    # 替换 -1 为 0.000001
    df.replace(-1, 0.000001, inplace=True)

    # 归一化其他特征
    def normalize_tag(tag, min_val, max_val):
        if min_val != max_val:
            return (tag - min_val) / (max_val - min_val)
        else:
            return 1.0

    def normalize_opacity(opacity):
        return opacity / 10

    def normalize_color(value):
        return (value / 360.0) * 3.5

    def normalize_stroke_width(stroke_width, min_val, max_val):
        if max_val != min_val:
            return (stroke_width - min_val) / (max_val - min_val)
        else:
            return 1.0

    def normalize_layer(layer):
        layer_list = eval(layer)
        # 如果数组为空，返回 0.0
        if not layer_list:
            return 0.0

        # 将第一个数字作为整数部分
        integer_part = layer_list[0]

        # 将后续的数字按小数位处理
        decimal_part = sum(val * (0.1 ** idx) for idx, val in enumerate(layer_list[1:], start=1))

        # 返回整数部分加上小数部分
        normalized_value = integer_part + decimal_part

        return normalized_value * 1

    def min_max_normalize(series):
        min_val = series.min()
        max_val = series.max()
        if min_val != max_val:
            return (series - min_val) / (max_val - min_val)
        else:
            return 1.0

    # 使用最大值归一化位置相关的特征，不再使用画布的长宽或面积
    position_columns = ['bbox_min_top', 'bbox_max_bottom', 'bbox_min_left', 'bbox_max_right', 'bbox_center_x',
                        'bbox_center_y', 'bbox_width', 'bbox_height']
    df[position_columns] = df[position_columns].apply(min_max_normalize, axis=0)

    # 使用最大值归一化面积相关特征
    area_columns = ['bbox_fill_area', 'bbox_stroke_area']
    df[area_columns] = df[area_columns].apply(min_max_normalize, axis=0) * 1.8

    # 归一化每一列
    df['tag'] = df['tag'].apply(lambda x: normalize_tag(x, df['tag'].min(), df['tag'].max())) * 0.6
    df['opacity'] = df['opacity'].apply(normalize_opacity) * 2
    df['fill_h'] = df['fill_h'].map(normalize_color)
    df['stroke_h'] = df['stroke_h'].map(normalize_color)
    df[['fill_s', 'fill_l', 'stroke_s', 'stroke_l']] = df[['fill_s', 'fill_l', 'stroke_s', 'stroke_l']] / 100.0 * 3
    df['stroke_width'] = df['stroke_width'].apply(
        lambda x: normalize_stroke_width(x, df['stroke_width'].min(), df['stroke_width'].max()))
    df['layer'] = df['layer'].apply(normalize_layer)

    # 保存归一化后的特征数据
    df.to_csv(output_path, index=False)

    # 左右翻转处理
    df_LR = df.copy()
    df_LR[['bbox_min_left', 'bbox_max_right']] = df_LR[['bbox_max_right', 'bbox_min_left']].values
    df_LR.to_csv(output_path_LR, index=False)

    # 上下翻转处理
    df_TB = df.copy()
    df_TB[['bbox_min_top', 'bbox_max_bottom']] = df_TB[['bbox_max_bottom', 'bbox_min_top']].values
    df_TB.to_csv(output_path_TB, index=False)
