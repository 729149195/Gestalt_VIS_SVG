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
    
    # 获取所有边界值的范围
    all_tops = [item['features'][10] for item in init_data]
    all_bottoms = [item['features'][11] for item in init_data]
    all_lefts = [item['features'][12] for item in init_data]
    all_rights = [item['features'][13] for item in init_data]
    
    # 计算区间范围
    top_intervals = generate_intervals(min(all_tops), max(all_tops), 8)
    bottom_intervals = generate_intervals(min(all_bottoms), max(all_bottoms), 7)
    left_intervals = generate_intervals(min(all_lefts), max(all_lefts), 8)
    right_intervals = generate_intervals(min(all_rights), max(all_rights), 8)
    
    # 处理每个元素
    for item in init_data:
        element_id = item['id'].split('/')[-1]
        tag_name = element_id.split('_')[0] if '_' in element_id else element_id
        
        # 处理位置数据
        process_position(item, tag_name, top_intervals, top_data, 10)
        process_position(item, tag_name, bottom_intervals, bottom_data, 11)
        process_position(item, tag_name, left_intervals, left_data, 12)
        process_position(item, tag_name, right_intervals, right_data, 13)
    
    # 处理颜色数据
    fill_colors = defaultdict(int)
    stroke_colors = defaultdict(int)
    
    for item in init_data:
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
    
    # 处理SVG属性数据
    attr_data = process_svg_attributes(svg_file_path)
    
    # 处理元素数量数据
    ele_data = process_element_numbers(svg_file_path)
    
    # 保存所有生成的JSON文件
    save_json_files({
        'Top_data.json': dict(top_data),
        'Bottom_data.json': dict(bottom_data),
        'Left_data.json': dict(left_data),
        'Right_data.json': dict(right_data),
        'fill_num.json': dict(fill_colors),
        'stroke_num.json': dict(stroke_colors),
        'attr_num.json': attr_data,
        'ele_num.json': ele_data
    }, output_dir)

def generate_intervals(min_val, max_val, num_intervals):
    """生成均匀的区间"""
    interval_size = (max_val - min_val) / num_intervals
    return [(min_val + i * interval_size, min_val + (i + 1) * interval_size) 
            for i in range(num_intervals)]

def process_position(item, tag_name, intervals, data_dict, feature_index):
    """处理位置数据"""
    value = item['features'][feature_index]
    for i, (start, end) in enumerate(intervals):
        if start <= value <= end:
            interval_key = f"{start:.1f}-{end:.1f}"
            data_dict[interval_key]["tags"].append(item['id'].split('/')[-1])
            tag_base = tag_name.split('_')[0]
            data_dict[interval_key]["total"][tag_base] += 1
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
