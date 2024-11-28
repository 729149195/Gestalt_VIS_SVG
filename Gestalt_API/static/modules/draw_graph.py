import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def calculate_distance(bbox1, bbox2):
    left1, right1, top1, bottom1 = bbox1['left'], bbox1['right'], bbox1['top'], bbox1['bottom']
    left2, right2, top2, bottom2 = bbox2['left'], bbox2['right'], bbox2['top'], bbox2['bottom']

    if right1 < left2:
        dx = left2 - right1
    elif right2 < left1:
        dx = left1 - right2
    else:
        dx = 0

    # 计算垂直距离
    if bottom1 < top2:
        dy = top2 - bottom1
    elif bottom2 < top1:
        dy = top1 - bottom2
    else:
        dy = 0

    distance = np.sqrt(dx ** 2 + dy ** 2)

    return distance


def draw_element_nodes_with_lines(init_json_path, cluster_probabilities_path):
    # 加载 init_json 数据
    with open(init_json_path, 'r') as init_file:
        init_data = json.load(init_file)

    # 加载 cluster_probabilities 数据
    with open(cluster_probabilities_path, 'r') as prob_file:
        cluster_probabilities = json.load(prob_file)

    fig, ax = plt.subplots()

    # 初始化变量
    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')
    nodes = []
    prob_dict = {item['id']: item['features'] for item in cluster_probabilities}

    # 处理每个元素
    for element in init_data:
        bbox_left = element['features'][12]
        bbox_right = element['features'][13]
        bbox_top = element['features'][10]
        bbox_bottom = element['features'][11]
        bbox_center_x = element['features'][14]
        bbox_center_y = element['features'][15]
        bbox_area = element['features'][18]
        element_id = element['id']
        nodes.append({
            'id': element_id,
            'left': bbox_left,
            'right': bbox_right,
            'top': bbox_top,
            'bottom': bbox_bottom,
            'center_x': bbox_center_x,
            'center_y': bbox_center_y,
            'area': bbox_area
        })

        # 更新最小和最大值
        min_x = min(min_x, bbox_left)
        max_x = max(max_x, bbox_right)
        min_y = min(min_y, bbox_top)
        max_y = max(max_y, bbox_bottom)

    # 设置绘图范围
    x_margin = (max_x - min_x) * 0.05
    y_margin = (max_y - min_y) * 0.05
    ax.set_xlim(min_x - x_margin, max_x + x_margin)
    ax.set_ylim(max_y + y_margin, min_y - y_margin)
    ax.set_aspect('equal', adjustable='box')

    # 绘制每个元素的边界框和中心点
    for node in nodes:
        bbox_left = node['left']
        bbox_right = node['right']
        bbox_top = node['top']
        bbox_bottom = node['bottom']
        bbox_center_x = node['center_x']
        bbox_center_y = node['center_y']
        bbox_area = node['area']

        rect_width = bbox_right - bbox_left
        rect_height = bbox_bottom - bbox_top
        rect = patches.Rectangle((bbox_left, bbox_top), rect_width, rect_height,
                                 linewidth=1, edgecolor='black', facecolor='none', alpha=0.5)
        ax.add_patch(rect)

        center_point_size = np.sqrt(bbox_area) / 10  # 调整这个比例因子以匹配面积
        ax.plot(bbox_center_x, bbox_center_y, 'ro', markersize=center_point_size)

    # 计算调整后的距离，并存储所有距离
    distances = []
    num_nodes = len(nodes)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            node1 = nodes[i]
            node2 = nodes[j]
            distance = calculate_distance(node1, node2)

            adjusted_distance = distance

            # 存储距离和中心点坐标
            distances.append(
                (adjusted_distance, node1['center_x'], node1['center_y'], node2['center_x'], node2['center_y']))

    # 对所有距离进行排序并保留前6%的最小调整后距离
    distances = sorted(distances, key=lambda x: x[0])
    top_6_percent_distances = distances[:int(len(distances) * 0.06)]

    # 提取调整后的距离用于归一化
    adjusted_distances = [d[0] for d in top_6_percent_distances]
    min_distance = min(adjusted_distances)
    max_distance = max(adjusted_distances)
    distance_range = max_distance - min_distance if max_distance != min_distance else 1

    # 定义不透明度范围
    min_alpha = 0.1
    max_alpha = 1.0  # 最大不透明度

    # # 绘制连线（使用距离），黑色表示距离
    # for adjusted_distance, x1, y1, x2, y2 in top_6_percent_distances:
    #     # 归一化并反转调整后的距离
    #     normalized_distance = (adjusted_distance - min_distance) / distance_range
    #     inverted_distance = 1 - normalized_distance
    #     alpha_value = min_alpha + (inverted_distance ** 1) * (max_alpha - min_alpha)
    #
    #     # 绘制连线（黑色线表示距离），线的粗细固定为1
    #     ax.plot([x1, x2], [y1, y2], 'k-', linewidth=1, alpha=alpha_value)

    # 计算节点之间的余弦相似度
    similarities = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            node1 = nodes[i]
            node2 = nodes[j]

            # 获取每个节点的概率
            prob1 = prob_dict.get(node1['id'])
            prob2 = prob_dict.get(node2['id'])
            if prob1 is not None and prob2 is not None:
                # 计算余弦相似度
                similarity = cosine_similarity([prob1], [prob2])[0][0]

                # 组合相似度和中心点坐标
                similarities.append((similarity, node1['center_x'], node1['center_y'], node2['center_x'], node2['center_y']))

    # 对相似度进行排序并保留前20%的最大相似度
    similarities = sorted(similarities, key=lambda x: -x[0])
    top_20_percent_similarities = similarities[:int(len(similarities) * 0.2)]

    # 提取相似度用于归一化
    similarities_scores = [s[0] for s in top_20_percent_similarities]
    min_similarity = min(similarities_scores)
    max_similarity = max(similarities_scores)
    similarity_range = max_similarity - min_similarity if max_similarity != min_similarity else 1

    # 绘制连线（使用相似度），蓝色表示相似性
    for similarity, x1, y1, x2, y2 in top_20_percent_similarities:
        # 归一化相似度
        normalized_similarity = (similarity - min_similarity) / similarity_range

        # 使用非线性映射（平方）使得不透明度差异更显著
        alpha_value = min_alpha + (normalized_similarity ** 1) * (max_alpha - min_alpha)

        # 绘制连线（蓝色线表示相似度），线的粗细固定为1
        ax.plot([x1, x2], [y1, y2], 'b-', linewidth=1, alpha=alpha_value)

    plt.show()
