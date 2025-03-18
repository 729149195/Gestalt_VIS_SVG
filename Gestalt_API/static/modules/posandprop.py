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
    
    # 确保有足够的数据点
    if len(all_tops) < 2:
        print("警告: 数据点不足，使用默认范围")
        min_val, max_val = 0, 100
        all_tops = [min_val, max_val]
        all_bottoms = [min_val, max_val]
        all_lefts = [min_val, max_val]
        all_rights = [min_val, max_val]
    
    # 计算区间范围
    top_intervals = generate_intervals(min(all_tops), max(all_tops), 8)
    bottom_intervals = generate_intervals(min(all_bottoms), max(all_bottoms), 7)
    left_intervals = generate_intervals(min(all_lefts), max(all_lefts), 8)
    right_intervals = generate_intervals(min(all_rights), max(all_rights), 8)
    
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
