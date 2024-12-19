import json
import lxml.etree as ET
import colorsys
from svgpath2mpl import parse_path as mpl_parse_path
import pandas as pd
from tqdm import tqdm
import re
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.path as mpath
import matplotlib.transforms as mtransforms

class SVGParser:
    def __init__(self, file_path):
        self.file_path = file_path
        self.existing_tags = {}

    @staticmethod
    def escape_text_content(svg_content):
        def replacer(match):
            text_with_tags = match.group(0)
            start_tag_end = text_with_tags.find('>') + 1
            end_tag_start = text_with_tags.rfind('<')
            text_content = text_with_tags[start_tag_end:end_tag_start]
            escaped_content = SVGParser.escape_special_xml_chars(text_content)
            return text_with_tags[:start_tag_end] + escaped_content + text_with_tags[end_tag_start:]

        return re.sub(r'<text[^>]*>.*?</text>', replacer, svg_content, flags=re.DOTALL)

    @staticmethod
    def escape_special_xml_chars(svg_content):
        svg_content = re.sub(r'&(?!(amp;|lt;|gt;|quot;|apos;))', '&amp;', svg_content)
        return svg_content

    @staticmethod
    def parse_svg(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            svg_content = file.read()
        svg_content = SVGParser.escape_text_content(svg_content)
        svg_content = re.sub(r'<\?xml.*?\?>', '', svg_content).encode('utf-8')
        tree = ET.ElementTree(ET.fromstring(svg_content))
        root = tree.getroot()
        return tree, root

    def extract_element_info(self, element):
        tag_with_namespace = element.tag
        tag_without_namespace = tag_with_namespace.split("}")[-1] if '}' in tag_with_namespace else tag_with_namespace

        if tag_without_namespace != "svg":
            count = self.existing_tags.get(tag_without_namespace, 0)
            full_tag = (
                f"{tag_without_namespace}_{count}"
                if count > 0
                else tag_without_namespace
            )
            self.existing_tags[tag_without_namespace] = count + 1
        else:
            full_tag = tag_without_namespace

        attributes = element.attrib
        text_content = element.text.strip() if element.text else None

        return full_tag, attributes, element.text

    def add_element_to_graph(self, element, parent_path='0', level=0, layer="0"):

        tag, attributes, text_content = self.extract_element_info(element)
        node_id = tag

        element.set('id', node_id)

        current_path = f"{parent_path}/{node_id}" if parent_path != '0' else node_id
        element.attrib['tag_name'] = current_path

        new_layer_counter = 0
        for child in reversed(element):
            child_layer = f"{layer}_{new_layer_counter}"
            self.add_element_to_graph(child, parent_path=current_path, level=level + 1, layer=child_layer)
            new_layer_counter += 1

    def build_graph(self, svg_root):
        self.add_element_to_graph(svg_root)

    def run(self):
        tree, svg_root = SVGParser.parse_svg(self.file_path)
        self.build_graph(svg_root)
        for elem in svg_root.iter():
            elem.tag = elem.tag.split('}', 1)[-1]
            attribs = list(elem.attrib.items())
            for k, v in attribs:
                if k.startswith('{'):
                    del elem.attrib[k]
                    elem.attrib[k.split('}', 1)[-1]] = v
        return tree


class LayerDataExtractor:
    def __init__(self):
        self.layer_structure = {"name": "0", "children": []}
        self.node_layers = {}

    def extract_layers(self, element, current_path='0'):
        element_id = element.attrib.get('id', None)
        if element_id:
            self.node_layers[element_id] = current_path.split('/')

        children = list(element)
        for index, child in enumerate(children):
            child_tag = child.tag.split('}')[-1]
            child_layer = f"{index}"
            child_path = f"{current_path}/{child_layer}"
            self.extract_layers(child, child_path)

    def get_node_layers(self):
        return self.node_layers


def svgid(svg_input_path, svg_output_path):
    parser = SVGParser(svg_input_path)
    svg_tree = parser.run()
    svg_tree.write(svg_output_path, encoding='utf-8', xml_declaration=True)


def get_color_features(color, current_color='black'):
    if color is None:
        return 0.0, 0.0, 0.0
    if color == 'currentColor':
        color = current_color
    if color.lower() == 'none':
        return -1.0, -1.0, -1.0

    try:
        if color.startswith('#'):
            color = color.lstrip('#')
            lv = len(color)
            rgb = tuple(int(color[i:i + lv // 3], 16) / 255.0 for i in range(0, lv, lv // 3))
        elif color.startswith('rgb'):
            rgb = tuple(map(int, re.findall(r'\d+', color)))
            rgb = (rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)
        elif color.startswith('hsl'):
            h, s, l = map(float, re.findall(r'[\d.]+', color))
            s /= 100.0
            l /= 100.0
            if l == 1.0:
                return h, 0.0, l * 100.0
            rgb = colorsys.hls_to_rgb(h / 360.0, l, s)
        else:
            rgb = mcolors.to_rgb(color)
    except ValueError:
        rgb = (0.0, 0.0, 0.0)

    if rgb == (0.0, 0.0, 0.0):
        return 0.0, 0.0, 0.0

    h, l, s = colorsys.rgb_to_hls(*rgb)
    return h * 360.0, s * 100.0, l * 100.0


def get_inherited_attribute(element, attribute_name):
    current_element = element
    while current_element is not None:
        if attribute_name in current_element.attrib:
            return current_element.attrib[attribute_name]
        current_element = current_element.getparent()
    return None


def apply_transform(transform_str, points):
    if not transform_str:
        return points
    # 初始化为单位矩阵
    transform_matrix = np.identity(3)
    transform_commands = re.findall(r'\w+\([^)]+\)', transform_str)
    for command in transform_commands:
        cmd_type = command.split('(')[0]
        values = list(map(float, re.findall(r'[-\d.]+', command)))
        if cmd_type == 'translate':
            dx, dy = values if len(values) == 2 else (values[0], 0)
            matrix = np.array([
                [1, 0, dx],
                [0, 1, dy],
                [0, 0, 1]
            ])
            transform_matrix = np.dot(transform_matrix, matrix)
        elif cmd_type == 'scale':
            if len(values) == 1:
                sx, sy = values[0], values[0]
            else:
                sx, sy = values
            matrix = np.array([
                [sx, 0, 0],
                [0, sy, 0],
                [0, 0, 1]
            ])
            transform_matrix = np.dot(transform_matrix, matrix)
        elif cmd_type == 'rotate':
            angle = np.radians(values[0])
            cos_val, sin_val = np.cos(angle), np.sin(angle)
            if len(values) == 3:
                cx, cy = values[1], values[2]
                matrix = np.array([
                    [cos_val, -sin_val, cx - cos_val * cx + sin_val * cy],
                    [sin_val, cos_val, cy - sin_val * cx - cos_val * cy],
                    [0, 0, 1]
                ])
            else:
                matrix = np.array([
                    [cos_val, -sin_val, 0],
                    [sin_val, cos_val, 0],
                    [0, 0, 1]
                ])
            transform_matrix = np.dot(transform_matrix, matrix)
        elif cmd_type == 'matrix':
            if len(values) == 6:
                a, b, c, d, e, f = values
                matrix = np.array([
                    [a, c, e],
                    [b, d, f],
                    [0, 0, 1]
                ])
                transform_matrix = np.dot(transform_matrix, matrix)
            else:
                # 无效的矩阵，跳过
                print(f"警告: 无效的 matrix 变换: {command}")
                continue
        else:
            # 未知的变换命令，跳过
            print(f"警告: 未知的变换命令: {cmd_type}")
            continue

    # 应用最终的变换矩阵到点上
    transformed_points = []
    for x, y in points:
        point = np.array([x, y, 1])
        transformed_point = np.dot(transform_matrix, point)
        transformed_points.append((transformed_point[0], transformed_point[1]))
    return transformed_points


def calculate_path_length(path):
    verts = path.vertices
    codes = path.codes
    length = 0
    for i in range(1, len(verts)):
        if codes[i] != mpath.Path.MOVETO:
            length += np.linalg.norm(verts[i] - verts[i - 1])
    return length


def apply_transform_to_path(path, transform_str):
    if not transform_str:
        return path.vertices

    transform = mtransforms.Affine2D()
    # 解析变换字符串并应用到 transform 对象
    transform_commands = re.findall(r'\w+\([^)]+\)', transform_str)
    for command in transform_commands:
        cmd_type = command.split('(')[0]
        values = list(map(float, re.findall(r'[-\d.]+', command)))
        if cmd_type == 'translate':
            dx, dy = values if len(values) == 2 else (values[0], 0)
            transform.translate(dx, dy)
        elif cmd_type == 'scale':
            if len(values) == 1:
                sx, sy = values[0], values[0]
            else:
                sx, sy = values
            transform.scale(sx, sy)
        elif cmd_type == 'rotate':
            angle = values[0]
            if len(values) == 3:
                cx, cy = values[1], values[2]
                transform.translate(cx, cy)
                transform.rotate_deg(angle)
                transform.translate(-cx, -cy)
            else:
                transform.rotate_deg(angle)
        # 处理其他变换类型（如 skewX, skewY）可以根据需要添加

    transformed_vertices = transform.transform(path.vertices)
    return transformed_vertices


def calculate_polygon_area(vertices):
    # 使用鞋带公式计算多边形面积
    x = vertices[:, 0]
    y = vertices[:, 1]
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    return area


def calculate_path_area(path):
    # 使用 matplotlib 的 Path 对象获取多边形并计算面积
    polys = path.to_polygons()
    total_area = 0.0
    for poly in polys:
        if len(poly) >= 3:
            poly = np.array(poly)
            area = calculate_polygon_area(poly)
            total_area += area
    return total_area


def is_visible(element):
    if element.attrib.get('display', '') == 'none':
        return False
    if element.attrib.get('visibility', '') == 'hidden':
        return False
    if float(element.attrib.get('opacity', 1.0)) == 0:
        return False
    if element.attrib.get('fill', 'currentColor').lower() == 'none' and element.attrib.get('stroke', 'currentColor').lower() == 'none':
        return False
    return True


def extract_features(element, layer_extractor, current_transform='', current_color='black'):
    # 过滤不处理的标签
    filter_tags = {'defs', 'symbol', 'clipPath', 'mask'}  # 根据需求调整过滤的标签
    tag_without_namespace = element.tag.split('}')[-1]
    # 如果元素的标签在过滤列表中，视为不可见元素
    if tag_without_namespace in filter_tags:
        return None

    # 如果元素不可见，则跳过
    if not is_visible(element):
        return None

    # 累积并传递父元素的 transform
    parent = element.getparent()
    while parent is not None:
        parent_transform = parent.attrib.get('transform', None)
        if parent_transform:
            current_transform = f"{parent_transform} {current_transform}"
        parent = parent.getparent()

    # 标签映射
    tag_mapping = {
    'rect': 0,
    'circle': 1,
    'ellipse': 2,
    'line': 3,
    'polyline': 4,
    'polygon': 5,
    'path': 6,
    'text': 7,
    'image': 8
}

    tag = element.tag.split('}')[-1]
    tag_value = tag_mapping.get(tag, 32)

    element_id = element.attrib.get('id', '0')
    element_id_number = re.findall(r'\d+', element_id)
    if element_id_number:
        element_id_number = element_id_number[0]
    else:
        element_id_number = '0'
    # tag_value = float(f"{tag_value}.{element_id_number:0>4}")

    opacity = float(element.attrib.get('opacity', 1.0))
    fill = element.attrib.get('fill', 'currentColor')
    stroke = element.attrib.get('stroke', 'currentColor')

    if fill == 'currentColor':
        fill = get_inherited_attribute(element, 'fill') or 'black'
    if stroke == 'currentColor':
        stroke = get_inherited_attribute(element, 'stroke') or 'none'
        
    stroke_width = 0.0
    if stroke.lower() != 'none':
        stroke_width = float(element.attrib.get('stroke-width', 1.0))

    fill_h, fill_s, fill_l = get_color_features(fill, current_color)
    stroke_h, stroke_s, stroke_l = get_color_features(stroke, current_color)

    # 当前元素的 transform 需要累积传递的 transform
    transform = element.attrib.get('transform', None)
    if transform:
        current_transform = f"{current_transform} {transform}"

    # 计算边界框
    bbox_values = get_transformed_bbox(element, current_transform)

    # 提取层级信息
    layer = layer_extractor.get_node_layers().get(element_id, ['0'])
    layer = [int(part.split('_')[-1]) if part != '0' else 0 for part in layer]

    tag_name = element.attrib.get('tag_name', '')

    return [
        tag_name, tag_value, opacity, fill_h, fill_s, fill_l,
        stroke_h, stroke_s, stroke_l, stroke_width,
        layer, *bbox_values
    ]


def get_transformed_bbox(element, current_transform=''):
    bbox = None
    fill_area = 0.0
    stroke_area = 0.0
    
    stroke = element.attrib.get('stroke', 'currentColor')
    if stroke == 'currentColor':
        stroke = get_inherited_attribute(element, 'stroke') or 'none'
    
    stroke_width = 0.0
    if stroke.lower() != 'none':
        stroke_width = float(element.attrib.get('stroke-width', 1.0))

    # 修改height属性的处理
    height_str = element.attrib.get('height', '0')
    height = 0 if height_str == 'auto' else float(height_str)
    
    # 同样处理width属性
    width_str = element.attrib.get('width', '0')
    width = 0 if width_str == 'auto' else float(width_str)
    
    if element.tag.endswith('rect'):
        x = float(element.attrib.get('x', 0))
        y = float(element.attrib.get('y', 0))
        bbox = [(x, y), (x + width, y), (x, y + height), (x + width, y + height)]
        fill_area = width * height
        stroke_area = 2 * (width + height) * stroke_width

    elif element.tag.endswith('circle'):
        cx = float(element.attrib.get('cx', 0))
        cy = float(element.attrib.get('cy', 0))
        r = float(element.attrib.get('r', 0))
        bbox = [(cx - r, cy - r), (cx + r, cy - r), (cx - r, cy + r), (cx + r, cy + r)]
        fill_area = np.pi * r * r
        stroke_area = 2 * np.pi * r * stroke_width

    elif element.tag.endswith('ellipse'):
        cx = float(element.attrib.get('cx', 0))
        cy = float(element.attrib.get('cy', 0))
        rx = float(element.attrib.get('rx', 0))
        ry = float(element.attrib.get('ry', 0))
        bbox = [(cx - rx, cy - ry), (cx + rx, cy - ry), (cx - rx, cy + ry), (cx + rx, cy + ry)]
        fill_area = np.pi * rx * ry
        stroke_area = 2 * np.pi * (rx + ry) * stroke_width

    elif element.tag.split('}')[-1] == 'line':
        x1 = float(element.attrib.get('x1', 0))
        y1 = float(element.attrib.get('y1', 0))
        x2 = float(element.attrib.get('x2', 0))
        y2 = float(element.attrib.get('y2', 0))
        half_stroke = stroke_width / 2
        bbox = [(x1 - half_stroke, y1 - half_stroke), (x2 + half_stroke, y2 + half_stroke),
                (x1 + half_stroke, y1 + half_stroke), (x2 - half_stroke, y2 - half_stroke)]
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        fill_area = 0.0
        stroke_area = length * stroke_width


    elif element.tag.split('}')[-1] == 'polyline' or element.tag.split('}')[-1] == 'polygon':

        points = element.attrib.get('points', '').strip().split()
        points = [tuple(map(float, point.split(','))) for point in points]
        if not points:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        xs, ys = zip(*points)
        bbox = [(min(xs), min(ys)), (max(xs), min(ys)), (min(xs), max(ys)), (max(xs), max(ys))]
        # 计算 polyline 或 polygon 的长度
        length = sum(np.sqrt((points[i + 1][0] - points[i][0]) ** 2 + (points[i + 1][1] - points[i][1]) ** 2) for i in
                     range(len(points) - 1))
        if element.tag.endswith('polygon'):
            # 如果是 polygon，则计算填充面积，并将最后一段的长度加入
            fill_area = 0.5 * np.abs(
                sum(points[i][0] * points[i + 1][1] - points[i + 1][0] * points[i][1] for i in range(len(points) - 1))
            )
            # 添加最后一个点与第一个点之间的长度
            length += np.sqrt((points[-1][0] - points[0][0]) ** 2 + (points[-1][1] - points[0][1]) ** 2)
        else:
            fill_area = 0.0  # polyline 没有填充面积
        stroke_area = length * stroke_width

    elif element.tag.endswith('path'):
        path_data = element.attrib.get('d', None)
        if path_data:
            path = mpl_parse_path(path_data)
            vertices = path.vertices
            xmin, ymin = vertices.min(axis=0)
            xmax, ymax = vertices.max(axis=0)
            bbox = [(xmin, ymin), (xmax, ymin), (xmin, ymax), (xmax, ymax)]
            try:
                if path.codes is not None and np.all(path.codes == 1):
                    fill_area = path.to_polygons()[0].area
                stroke_area = calculate_path_length(path) * stroke_width
            except AssertionError:
                stroke_area = calculate_path_length(path) * stroke_width

    # 在 get_transformed_bbox 函数的 text 处理部分使用 parse_font_size 函数
    elif element.tag.endswith('text'):
        text_content = element.text or ''
        bbox = [(0, 0), (0, 0), (0, 0), (0, 0)]
        fill_area = 0.0
        stroke_area = 0.0
        if text_content.strip():
            font_size = parse_font_size(element.attrib.get('font-size', '16px'))
            x = float(element.attrib.get('x', 0))
            y = float(element.attrib.get('y', 0))
            text_width = len(text_content) * font_size * 0.6  # 文本宽度
            bbox = [(x - text_width / 2, y - font_size),     # 左上角
                    (x + text_width / 2, y - font_size),     # 右上角
                    (x - text_width / 2, y),                 # 左下角
                    (x + text_width / 2, y)]                 # 右下角
            fill_area = text_width * font_size

    if bbox:
        transform = element.attrib.get('transform', None)
        if transform:
            current_transform = f"{current_transform} {transform}"
        transformed_points = apply_transform(current_transform, bbox)
        xs, ys = zip(*transformed_points)
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        width = xmax - xmin
        height = ymax - ymin
        return ymin, ymax, xmin, xmax, (xmin + xmax) / 2, (ymin + ymax) / 2, width, height, fill_area, stroke_area

    return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0


def parse_font_size(font_size_str):
    # 默认的 font-size，若未设置则默认为 16 px
    default_font_size = 16.0
    if isinstance(font_size_str, (float, int)):
        return float(font_size_str)  # 无单位的情况
    if font_size_str.endswith('px'):
        return float(font_size_str.replace('px', ''))
    elif font_size_str.endswith('pt'):
        # 将 pt 转换为 px，1 pt ≈ 1.3333 px
        return float(font_size_str.replace('pt', '')) * 1.3333
    elif font_size_str.endswith('em'):
        # em 基于默认 font-size
        return float(font_size_str.replace('em', '')) * default_font_size
    elif font_size_str.endswith('%'):
        # 百分比相对于默认 font-size
        return float(font_size_str.replace('%', '')) * default_font_size / 100.0
    else:
        # 若无单位或无法识别单位，使用默认 font-size
        try:
            return float(font_size_str)
        except ValueError:
            return default_font_size


def process_svg(file_path):
    svg_parser = SVGParser(file_path)
    tree = svg_parser.run()
    root = tree.getroot()

    # 提取 viewBox 和 width/height 属性
    viewBox = root.attrib.get('viewBox', None)
    width = root.attrib.get('width', None)
    height = root.attrib.get('height', None)

    initial_transform = ''
    if viewBox and width and height:
        # 使用正则表达式分割 viewBox，以处理逗号和空白字符
        numbers = re.split(r'[,\s]+', viewBox.strip())
        if len(numbers) != 4:
            print(f"警告: viewBox 的格式不正确: {viewBox}")
            minX, minY, widthV, heightV = 0.0, 0.0, float(width), float(height)
        else:
            try:
                minX, minY, widthV, heightV = map(float, numbers)
            except ValueError:
                print(f"警告: 无法将 viewBox 数值转换为浮点数: {viewBox}")
                minX, minY, widthV, heightV = 0.0, 0.0, float(width), float(height)

        try:
            width = float(width)
            height = float(height)
            sx = width / widthV
            sy = height / heightV
            tx = -minX * sx
            ty = -minY * sy
            # 构建矩阵变换
            initial_transform = f"matrix({sx}, 0, 0, {sy}, {tx}, {ty})"
        except Exception as e:
            # print(f"错误: 计算 viewBox 变换时出错: {e}")
            initial_transform = ''

    layer_extractor = LayerDataExtractor()
    layer_extractor.extract_layers(root)

    features = []
    elements = list(root.iter())
    for element in tqdm(elements, total=len(elements), desc="Processing SVG Elements"):
        tag = element.tag.split('}')[-1]
        if tag in {'circle', 'rect', 'line', 'polyline', 'polygon', 'path', 'text', 'ellipse', 'image', 'use'}:
            feature = extract_features(element, layer_extractor, current_transform=initial_transform)
            if feature:
                features.append(feature)
    return features


def save_features(features, output_path):
    columns = [
        'tag_name', 'tag', 'opacity', 'fill_h', 'fill_s', 'fill_l',
        'stroke_h', 'stroke_s', 'stroke_l', 'stroke_width',
        'layer', 'bbox_min_top', 'bbox_max_bottom', 'bbox_min_left', 'bbox_max_right',
        'bbox_center_x', 'bbox_center_y', 'bbox_width', 'bbox_height', 'bbox_fill_area', 'bbox_stroke_area'
    ]
    df = pd.DataFrame(features, columns=columns)
    df.to_csv(output_path, index=False)

def process_csv_to_json(input_csv_path, output_json_path):
    df = pd.read_csv(input_csv_path)
    json_data = []
    for index, row in df.iterrows():
        element_id = row['tag_name']
        features = row[1:].tolist()

        json_data.append({
            "id": element_id,
            "features": features
        })
    with open(output_json_path, 'w') as json_file:
        json.dump(json_data, json_file, indent=4)


def save_svg_with_ids(svg_input_path, svg_output_path):
    svgid(svg_input_path, svg_output_path)


def process_and_save_features(svg_input_path, output_csv_path, output_svg_with_ids_path):
    features = process_svg(svg_input_path)
    save_features(features, output_csv_path)
    save_svg_with_ids(svg_input_path, output_svg_with_ids_path)
