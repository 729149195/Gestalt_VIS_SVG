import json
import os
from bs4 import BeautifulSoup
from collections import defaultdict, Counter

def process_position_and_properties(init_json_path, svg_file_path, output_dir):
    """处理位置和属性信息,生成所需的JSON文件"""
    
    # 读取init_json文件
    with open(init_json_path, 'r', encoding='utf-8') as f:
        init_data = json.load(f)
    
    # 处理位置数据
    top_data = defaultdict(lambda: {"tags": [], "total": defaultdict(int)})
    bottom_data = defaultdict(lambda: {"tags": [], "total": defaultdict(int)})
    left_data = defaultdict(lambda: {"tags": [], "total": defaultdict(int)})
    right_data = defaultdict(lambda: {"tags": [], "total": defaultdict(int)})
    # 添加width和height数据结构
    width_data = defaultdict(lambda: {"tags": [], "total": defaultdict(int)})
    height_data = defaultdict(lambda: {"tags": [], "total": defaultdict(int)})
    
    # 获取过滤后SVG中的所有元素ID
    filtered_element_ids = set()
    try:
        with open(svg_file_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), 'xml')
            # 获取所有带有ID的元素
            for element in soup.find_all(lambda tag: tag.get('id')):
                filtered_element_ids.add(element.get('id'))
            
            # 获取所有带有tag_name的元素
            for element in soup.find_all(lambda tag: tag.get('tag_name')):
                filtered_element_ids.add(element.get('tag_name'))
        
        print(f"过滤后的SVG文件中找到 {len(filtered_element_ids)} 个元素")
    except Exception as e:
        print(f"读取过滤后的SVG文件时出错: {str(e)}")
        filtered_element_ids = set()  # 如果出错，使用空集合

    # 只处理在过滤后SVG中存在的元素
    filtered_init_data = []
    for item in init_data:
        element_id = item['id']
        # 检查元素是否在过滤后的SVG中
        if not filtered_element_ids:
            filtered_init_data.append(item)
        elif isinstance(element_id, str):
            # 只有当id是字符串时才尝试split操作
            if element_id in filtered_element_ids or element_id.split('/')[-1] in filtered_element_ids:
                filtered_init_data.append(item)
        else:
            # 处理id不是字符串的情况
            if element_id in filtered_element_ids:
                filtered_init_data.append(item)
    
    # 如果没有找到匹配的元素，使用所有元素（回退策略）
    if not filtered_init_data and init_data:
        print("警告: 没有找到与过滤后SVG匹配的元素，将使用所有元素")
        filtered_init_data = init_data
    
    print(f"处理 {len(filtered_init_data)} 个元素（共 {len(init_data)} 个）")
    
    # 获取过滤后元素的边界值范围
    all_tops = [item['features'][10] for item in filtered_init_data]
    all_bottoms = [item['features'][11] for item in filtered_init_data]
    all_lefts = [item['features'][12] for item in filtered_init_data]
    all_rights = [item['features'][13] for item in filtered_init_data]
    # 获取width和height的值范围
    all_widths = [item['features'][16] for item in filtered_init_data]
    all_heights = [item['features'][17] for item in filtered_init_data]
    
    # 确保有足够的数据点
    if len(all_tops) < 2:
        print("警告: 数据点不足，使用默认范围")
        min_val, max_val = 0, 100
        all_tops = [min_val, max_val]
        all_bottoms = [min_val, max_val]
        all_lefts = [min_val, max_val]
        all_rights = [min_val, max_val]
        all_widths = [min_val, max_val]
        all_heights = [min_val, max_val]
    
    # 计算区间范围 - 减少区间数量并优化分布
    # 使用更少的区间数量，通常4个区间对于大多数可视化效果已足够
    top_intervals = generate_optimized_intervals(all_tops, 6)
    bottom_intervals = generate_optimized_intervals(all_bottoms, 6)
    left_intervals = generate_optimized_intervals(all_lefts, 6)
    right_intervals = generate_optimized_intervals(all_rights, 6)
    # 计算width和height的区间范围
    width_intervals = generate_optimized_intervals(all_widths, 6)
    height_intervals = generate_optimized_intervals(all_heights, 6)
    
    # 处理每个元素
    for item in filtered_init_data:
        # 从id中获取元素标识符，在init_data中，id通常是从tag_name转换过来的
        if isinstance(item['id'], str):
            element_id = item['id'].split('/')[-1]
        else:
            # 如果id不是字符串类型，则直接使用该值作为id
            element_id = str(item['id'])
        
        # 避免依赖于"tagname_number"格式
        # 尝试从SVG文件中获取元素的类型标识，或直接使用元素ID
        # 如果ID中包含下划线，尝试提取标签类型，但不假设特定格式
        if '_' in element_id and any(element_id.startswith(tag) for tag in ['rect', 'circle', 'ellipse', 'path', 'line', 'polygon', 'polyline', 'text', 'image']):
            tag_name = element_id.split('_')[0]
        else:
            # 如果不是SVGParser生成的ID格式，则使用整个ID
            tag_name = element_id
        
        # 处理位置数据
        process_position(item, tag_name, top_intervals, top_data, 10)
        process_position(item, tag_name, bottom_intervals, bottom_data, 11)
        process_position(item, tag_name, left_intervals, left_data, 12)
        process_position(item, tag_name, right_intervals, right_data, 13)
        # 处理width和height数据
        process_position(item, tag_name, width_intervals, width_data, 16)
        process_position(item, tag_name, height_intervals, height_data, 17)
    
    # 处理颜色数据
    fill_colors = defaultdict(int)
    stroke_colors = defaultdict(int)
    
    for item in filtered_init_data:
        # 处理填充颜色 (h,s,l 分别在索引 2,3,4)
        h_fill, s_fill, l_fill = item['features'][2:5]
        if h_fill != -1 and s_fill != -1 and l_fill != -1:
            color_key = f"hsl({h_fill:.1f}, {s_fill:.1f}%, {l_fill:.1f}%)"
            fill_colors[color_key] += 1
            
        # 处理边框颜色 (h,s,l 分别在索引 5,6,7)
        h_stroke, s_stroke, l_stroke = item['features'][5:8]
        if h_stroke != -1 and s_stroke != -1 and l_stroke != -1:
            color_key = f"hsl({h_stroke:.1f}, {s_stroke:.1f}%, {l_stroke:.1f}%)"
            stroke_colors[color_key] += 1
    
    # 处理SVG属性数据 - 使用过滤后的SVG文件
    attr_data = process_svg_attributes(svg_file_path)
    
    # 处理元素数量数据 - 使用过滤后的SVG文件
    ele_data = process_element_numbers(svg_file_path)
    
    # 保存所有生成的JSON文件
    save_json_files({
        'Top_data.json': dict(top_data),
        'Bottom_data.json': dict(bottom_data),
        'Left_data.json': dict(left_data),
        'Right_data.json': dict(right_data),
        'Width_data.json': dict(width_data),  # 添加width数据
        'Height_data.json': dict(height_data),  # 添加height数据
        'fill_num.json': dict(fill_colors),
        'stroke_num.json': dict(stroke_colors),
        'attr_num.json': attr_data,
        'ele_num.json': ele_data
    }, output_dir)

def generate_optimized_intervals(values, num_intervals=6):
    """生成优化的区间划分，确保数据分布更加集中
    
    参数:
        values: 数据值列表
        num_intervals: 需要划分的区间数量，默认为6
    返回:
        区间列表，每个区间为(start, end)元组
    """
    if not values:
        return []
    
    # 对值进行排序
    sorted_values = sorted(values)
    n = len(sorted_values)
    
    # 如果数据点少于区间数，则使用特殊处理
    if n <= num_intervals:
        # 针对每个数据点创建一个小区间
        intervals = []
        min_val, max_val = min(sorted_values), max(sorted_values)
        range_size = max(0.001, (max_val - min_val) * 0.1)  # 确保区间有一定宽度
        
        for val in sorted_values:
            intervals.append((val - range_size/2, val + range_size/2))
        
        # 如果需要更多区间，则添加空区间
        while len(intervals) < num_intervals:
            intervals.append((max_val, max_val + range_size))
            max_val += range_size
        
        return intervals
    
    # 使用自适应聚类方法生成区间
    # 尝试K-means类似的简化算法来识别数据中的自然聚类
    
    # 步骤1：计算数据的基本统计信息
    min_val = sorted_values[0]
    max_val = sorted_values[-1]
    data_range = max_val - min_val
    
    # 如果数据几乎是均匀分布的（方差很小），使用等宽划分
    if data_range < 1e-10:  # 避免除零错误
        return [(min_val, max_val)]
    
    # 步骤2：尝试使用基于数据密度的方法识别聚类
    # 首先计算点之间的距离
    distances = [sorted_values[i+1] - sorted_values[i] for i in range(n-1)]
    
    # 如果没有距离（只有一个数据点），返回单个区间
    if not distances:
        return [(min_val, min_val + 0.001)]
    
    avg_distance = sum(distances) / len(distances)
    
    # 找出显著大于平均距离的间隔
    # 使用自适应阈值，根据数据分布特性调整
    threshold_multiplier = 1.5  # 可以调整这个值来控制聚类的灵敏度
    significant_gaps = []
    
    # 首先尝试较高的阈值
    for multiplier in [2.0, 1.75, 1.5, 1.25, 1.0]:
        significant_gaps = [(i, dist) for i, dist in enumerate(distances) 
                           if dist > avg_distance * multiplier]
        # 如果找到足够的间隔，跳出循环
        if len(significant_gaps) >= num_intervals - 1 and len(significant_gaps) <= num_intervals + 1:
            break
    
    # 如果找到了合适的间隔，使用它们来划分数据
    if significant_gaps and len(significant_gaps) >= 2:
        # 选择最显著的间隔
        significant_gaps.sort(key=lambda x: -x[1])
        target_gaps = min(len(significant_gaps), num_intervals - 1)
        gap_indices = sorted([gap[0] for gap in significant_gaps[:target_gaps]])
        
        # 根据间隔构建区间
        intervals = []
        start_idx = 0
        
        for gap_idx in gap_indices:
            end_idx = gap_idx + 1
            intervals.append((sorted_values[start_idx], sorted_values[end_idx-1]))
            start_idx = end_idx
        
        # 添加最后一个区间
        intervals.append((sorted_values[start_idx], sorted_values[-1]))
        
        # 如果区间数量不足，分割最大的区间
        while len(intervals) < num_intervals:
            # 找出最宽的区间
            widest_interval_idx = max(range(len(intervals)), 
                                      key=lambda i: intervals[i][1] - intervals[i][0])
            widest_interval = intervals.pop(widest_interval_idx)
            mid_point = (widest_interval[0] + widest_interval[1]) / 2
            intervals.append((widest_interval[0], mid_point))
            intervals.append((mid_point, widest_interval[1]))
            # 重新排序区间
            intervals.sort(key=lambda x: x[0])
        
        # 确保区间不重叠
        adjusted_intervals = []
        for i, (start, end) in enumerate(intervals):
            if i > 0:
                prev_end = adjusted_intervals[-1][1]
                if start <= prev_end:
                    start = prev_end + avg_distance * 0.01
            adjusted_intervals.append((start, end))
        
        return adjusted_intervals[:num_intervals]
    
    # 步骤3：回退方法 - 使用分位数
    # 这种方法适合于均匀分布的数据
    # 尝试将数据划分为大致相等大小的组，但确保相近的点在同一组
    
    # 首先，尝试检测数据中的"自然分组"
    groups = []
    current_group = [sorted_values[0]]
    
    for i in range(1, n):
        curr_val = sorted_values[i]
        prev_val = sorted_values[i-1]
        
        # 如果当前值与前一个值非常接近，它们属于同一组
        if curr_val - prev_val < avg_distance * 0.5:
            current_group.append(curr_val)
        else:
            # 否则，开始一个新组
            groups.append(current_group)
            current_group = [curr_val]
    
    # 添加最后一个组
    if current_group:
        groups.append(current_group)
    
    # 如果组的数量与目标区间数量接近，直接使用它们
    if len(groups) >= num_intervals * 0.75 and len(groups) <= num_intervals * 1.5:
        intervals = []
        
        # 合并小组，直到剩下num_intervals个组
        while len(groups) > num_intervals:
            # 找出最小的两个相邻组
            min_size_sum = float('inf')
            merge_idx = 0
            
            for i in range(len(groups) - 1):
                size_sum = len(groups[i]) + len(groups[i+1])
                if size_sum < min_size_sum:
                    min_size_sum = size_sum
                    merge_idx = i
            
            # 合并这两个组
            groups[merge_idx].extend(groups[merge_idx+1])
            groups.pop(merge_idx+1)
        
        # 拆分大组，直到有num_intervals个组
        while len(groups) < num_intervals:
            # 找出最大的组
            max_idx = max(range(len(groups)), key=lambda i: len(groups[i]))
            
            if len(groups[max_idx]) <= 1:
                # 如果最大的组只有一个元素，无法再拆分
                break
                
            # 将组拆分为两部分
            split_point = len(groups[max_idx]) // 2
            new_group = groups[max_idx][split_point:]
            groups[max_idx] = groups[max_idx][:split_point]
            groups.append(new_group)
        
        # 从组创建区间
        for group in groups:
            if group:  # 确保组非空
                intervals.append((min(group), max(group)))
        
        # 排序区间
        intervals.sort(key=lambda x: x[0])
        
        # 调整区间边界，确保不重叠
        adjusted_intervals = []
        for i, (start, end) in enumerate(intervals):
            if i > 0:
                prev_end = adjusted_intervals[-1][1]
                if start <= prev_end:
                    start = prev_end + avg_distance * 0.01
            
            # 确保每个区间有一定宽度
            if start >= end:
                end = start + avg_distance * 0.01
                
            adjusted_intervals.append((start, end))
        
        # 如果调整后的区间数量仍然过多，合并一些
        while len(adjusted_intervals) > num_intervals:
            # 找出最接近的两个区间
            min_gap = float('inf')
            merge_idx = 0
            
            for i in range(len(adjusted_intervals) - 1):
                gap = adjusted_intervals[i+1][0] - adjusted_intervals[i][1]
                if gap < min_gap:
                    min_gap = gap
                    merge_idx = i
            
            # 合并这两个区间
            new_interval = (adjusted_intervals[merge_idx][0], adjusted_intervals[merge_idx+1][1])
            adjusted_intervals[merge_idx] = new_interval
            adjusted_intervals.pop(merge_idx+1)
        
        if adjusted_intervals:
            return adjusted_intervals
    
    # 如果以上方法都失败，回退到基本的等分位数方法
    # 确保每个区间包含大致相同数量的数据点
    item_per_interval = n / num_intervals
    intervals = []
    
    for i in range(num_intervals):
        start_idx = min(n-1, int(i * item_per_interval))
        end_idx = min(n-1, int((i+1) * item_per_interval) - 1)
        
        # 确保最后一个区间包含最后一个数据点
        if i == num_intervals - 1:
            end_idx = n - 1
        
        start_val = sorted_values[start_idx]
        end_val = sorted_values[end_idx]
        
        # 确保区间有一定宽度
        if start_val == end_val:
            epsilon = max(0.001, (sorted_values[-1] - sorted_values[0]) * 0.01)
            end_val = start_val + epsilon
        
        intervals.append((start_val, end_val))
    
    # 确保连续区间边界不重叠
    final_intervals = []
    for i, (start, end) in enumerate(intervals):
        if i > 0:
            prev_end = final_intervals[-1][1]
            if start <= prev_end:
                # 如果有重叠，略微调整当前区间的起始值
                start = prev_end + (sorted_values[-1] - sorted_values[0]) * 0.001
        
        # 确保最后一个区间覆盖到最大值
        if i == len(intervals) - 1:
            end = max(end, sorted_values[-1] + (sorted_values[-1] - sorted_values[0]) * 0.001)
        
        final_intervals.append((start, end))
    
    return final_intervals

def process_position(item, tag_name, intervals, data_dict, feature_index):
    """处理位置数据"""
    value = item['features'][feature_index]
    for i, (start, end) in enumerate(intervals):
        if start <= value <= end:
            interval_key = f"{start:.1f}-{end:.1f}"
            # 使用完整的元素ID
            if isinstance(item['id'], str):
                element_id = item['id'].split('/')[-1]
            else:
                element_id = str(item['id'])
            data_dict[interval_key]["tags"].append(element_id)
            
            # 使用传入的tag_name，不做二次处理
            data_dict[interval_key]["total"][tag_name] += 1
            break

def process_svg_attributes(svg_file_path):
    """处理SVG属性数据"""
    with open(svg_file_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f.read(), 'xml')
    
    attr_counter = Counter()
    for element in soup.find_all():
        for attr in element.attrs:
            attr_counter[attr] += 1
    
    return [{"attribute": attr, "num": count} 
            for attr, count in attr_counter.most_common()]

def process_element_numbers(svg_file_path):
    """处理元素数量数据"""
    with open(svg_file_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f.read(), 'xml')
    
    visible_elements = {'rect', 'circle', 'ellipse', 'line', 
                       'polyline', 'polygon', 'path', 'text', 
                       'image', 'a'}
    
    element_counts = []
    for tag in soup.find_all():
        if not tag.name:
            continue
        
        element_counts.append({
            "tag": tag.name,
            "num": len(soup.find_all(tag.name)),
            "visible": tag.name in visible_elements
        })
    
    # 去重并按数量排序
    unique_counts = {}
    for item in element_counts:
        tag = item["tag"]
        if tag not in unique_counts:
            unique_counts[tag] = item
    
    return sorted(unique_counts.values(), 
                 key=lambda x: (-x["num"], x["tag"]))

def save_json_files(data_dict, output_dir):
    """保存所有JSON文件"""
    for filename, data in data_dict.items():
        output_path = os.path.join(output_dir, filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
