import xml.etree.ElementTree as ET
from collections import Counter
import re

def parse_svg(svg_file):
    """解析SVG文件，提取所有元素及其属性"""
    tree = ET.parse(svg_file)
    root = tree.getroot()
    elements = []
    for elem in root.iter():
        tag = elem.tag.split('}')[-1]  # 去除命名空间
        attrs = elem.attrib
        if tag == 'rect':
            elements.append({'type': 'rect', 'x': float(attrs.get('x', 0)), 'y': float(attrs.get('y', 0)),
                             'width': float(attrs.get('width', 0)), 'height': float(attrs.get('height', 0)),
                             'fill': attrs.get('fill')})
        elif tag == 'circle':
            elements.append({'type': 'circle', 'cx': float(attrs.get('cx', 0)), 'cy': float(attrs.get('cy', 0)),
                             'r': float(attrs.get('r', 0)), 'fill': attrs.get('fill')})
        elif tag == 'line':
            elements.append({'type': 'line', 'x1': float(attrs.get('x1', 0)), 'y1': float(attrs.get('y1', 0)),
                             'x2': float(attrs.get('x2', 0)), 'y2': float(attrs.get('y2', 0)),
                             'stroke': attrs.get('stroke')})
        elif tag == 'path':
            elements.append({'type': 'path', 'd': attrs.get('d', ''), 'fill': attrs.get('fill'),
                             'stroke': attrs.get('stroke')})
    return elements

def is_aligned(elements, axis='x'):
    """检查元素是否沿某轴对齐"""
    coords = [e[axis] for e in elements]
    diffs = [abs(coords[i+1] - coords[i]) for i in range(len(coords)-1)]
    return len(set(diffs)) <= 2  # 允许小误差

def is_grid(elements):
    """检查是否为网格排列"""
    xs = sorted(set(e['x'] for e in elements))
    ys = sorted(set(e['y'] for e in elements))
    return len(xs) > 1 and len(ys) > 1 and all(abs(xs[i+1] - xs[i]) < 5 for i in range(len(xs)-1))

def is_sector(d):
    """检查路径是否定义扇形"""
    return 'A' in d and 'L' in d  # 弧形和直线组合

def classify_chart(elements):
    """根据元素特征分类图表"""
    if not elements:
        return 'unknown'
    
    # 统计元素类型
    type_counts = Counter(e['type'] for e in elements)
    rects = [e for e in elements if e['type'] == 'rect']
    circles = [e for e in elements if e['type'] == 'circle']
    lines = [e for e in elements if e['type'] == 'line']
    paths = [e for e in elements if e['type'] == 'path']

    # 决策树分类
    if type_counts['rect'] > 5:  # 矩形为主
        if is_aligned(rects, 'x') or is_aligned(rects, 'y'):
            return 'bar_chart'
        if is_grid(rects):
            return 'heatmap'
        if any(r1['x'] > r2['x'] and r1['y'] > r2['y'] for r1 in rects for r2 in rects if r1 != r2):
            return 'treemap'
    
    if type_counts['circle'] > 5:  # 圆形为主
        if type_counts['line'] > 0:
            return 'network'
        return 'scatter'
    
    if type_counts['line'] > 5:  # 线条为主
        return 'line_chart'
    
    if type_counts['path'] > 0:  # 路径为主
        if any(is_sector(p['d']) for p in paths):
            return 'pie_chart'
        if any(p['fill'] and 'M' in p['d'] and 'Z' in p['d'] for p in paths):
            return 'area_chart'
        if type_counts['rect'] > 0:
            return 'sankey'
        return 'map'
    
    if type_counts['rect'] > 0 and type_counts['line'] > 0:
        return 'other_statistical'  # 箱线图等
    
    if len(set(type_counts.keys())) > 2:
        return 'mixed'
    
    return 'unknown'

# 使用示例
svg_file = '4.svg'
elements = parse_svg(svg_file)
chart_type = classify_chart(elements)
print(f'图表大类为：{chart_type}')