import json
import lxml.etree as ET
import colorsys
from svgpath2mpl import parse_path as mpl_parse_path
import pandas as pd
from tqdm import tqdm
import re
import numpy as np
import matplotlib.colors as mcolors


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

        if text_content:
            element.text = 'x' * len(text_content)

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
    if transform_str is None:
        return points
    transform_commands = re.findall(r'\w+\([^)]+\)', transform_str)
    for command in transform_commands:
        cmd_type = command.split('(')[0]
        values = list(map(float, re.findall(r'[-\d.]+', command)))
        if cmd_type == 'translate':
            dx, dy = values if len(values) == 2 else (values[0], 0)
            points = [(x + dx, y + dy) for x, y in points]
        elif cmd_type == 'scale':
            if len(values) == 1:
                sx, sy = values[0], values[0]
            else:
                sx, sy = values
            points = [(x * sx, y * sy) for x, y in points]
        elif cmd_type == 'rotate':
            angle = np.radians(values[0])
            cos_val, sin_val = np.cos(angle), np.sin(angle)
            if len(values) == 3:
                cx, cy = values[1], values[2]
                points = [(cos_val * (x - cx) - sin_val * (y - cy) + cx,
                           sin_val * (x - cx) + cos_val * (y - cy) + cy) for x, y in points]
            else:
                points = [(x * cos_val - y * sin_val, x * sin_val + y * cos_val) for x, y in points]
        # Add more transform types as needed
    return points


def calculate_path_length(path):
    verts = path.vertices
    codes = path.codes
    length = 0
    for i in range(1, len(verts)):
        if codes[i] != 1:  # If not a MOVETO
            length += np.linalg.norm(verts[i] - verts[i - 1])
    return length


def get_transformed_bbox(element, current_transform=''):
    bbox = None
    fill_area = 0.0
    stroke_area = 0.0
    stroke_width = float(element.attrib.get('stroke-width', 1.0))

    if element.tag.endswith('rect'):
        x = float(element.attrib.get('x', 0))
        y = float(element.attrib.get('y', 0))
        width = float(element.attrib.get('width', 0))
        height = float(element.attrib.get('height', 0))
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

    elif element.tag.endswith('line'):
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

    elif element.tag.endswith('polyline') or element.tag.endswith('polygon'):
        points = element.attrib.get('points', '').strip().split()
        points = [tuple(map(float, point.split(','))) for point in points]
        xs, ys = zip(*points)
        bbox = [(min(xs), min(ys)), (max(xs), min(ys)), (min(xs), max(ys)), (max(xs), max(ys))]
        length = sum(np.sqrt((points[i + 1][0] - points[i][0]) ** 2 + (points[i + 1][1] - points[i][1]) ** 2) for i in
                     range(len(points) - 1))
        if element.tag.endswith('polygon'):
            fill_area = 0.5 * np.abs(
                sum(points[i][0] * points[i + 1][1] - points[i + 1][0] * points[i][1] for i in range(len(points) - 1)))
            length += np.sqrt((points[-1][0] - points[0][0]) ** 2 + (points[-1][1] - points[0][1]) ** 2)
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
                    # print(fill_area)
                stroke_area = calculate_path_length(path) * stroke_width
            except AssertionError:
                stroke_area = calculate_path_length(path) * stroke_width

    elif element.tag.endswith('text'):
        text_content = element.text or ''
        bbox = [(0, 0), (0, 0), (0, 0), (0, 0)]
        fill_area = 0.0
        stroke_area = 0.0
        if text_content.strip():
            font_size = float(element.attrib.get('font-size', 16))
            x = float(element.attrib.get('x', 0))
            y = float(element.attrib.get('y', 0))
            bbox = [(x, y - font_size), (x + len(text_content) * font_size * 0.6, y - font_size),
                    (x, y), (x + len(text_content) * font_size * 0.6, y)]
            fill_area = (bbox[1][0] - bbox[0][0]) * (bbox[2][1] - bbox[0][1])

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


def is_visible(element):
    if element.attrib.get('display', '') == 'none':
        return False
    if element.attrib.get('visibility', '') == 'hidden':
        return False
    if float(element.attrib.get('opacity', 1.0)) == 0:
        return False
    if element.attrib.get('fill', 'currentColor').lower() == 'none' and element.attrib.get('stroke',
                                                                                           'currentColor').lower() == 'none':
        return False
    return True


def extract_features(element, layer_extractor, current_transform='', current_color='black'):
    #元素过滤
    filter_tags = {'g', 'defs','',''}  # 你可以根据需要调整这个列表
    tag_without_namespace = element.tag.split('}')[-1]
    # 如果元素的标签在过滤列表中，视为不可见元素
    if tag_without_namespace in filter_tags:
        return None

    if not is_visible(element):
        return None

    tag_mapping = {
        'circle': 1, 'rect': 4, 'line': 6, 'polyline': 8, 'polygon': 10,
        'path': 12, 'text': 14, 'g': 19, 'ellipse': 18, 'image': 20, 'use': 22,
        'defs': 24, 'linearGradient': 26, 'radialGradient': 28, 'stop': 30,
        'symbol': 33, 'clipPath': 34, 'mask': 36, 'pattern': 38, 'filter': 40,
        'feGaussianBlur': 42, 'feOffset': 44, 'feBlend': 46, 'feFlood': 48,
        'feImage': 50, 'feComposite': 52, 'feColorMatrix': 54, 'feMerge': 56,
        'feMorphology': 58, 'feTurbulence': 60, 'feDisplacementMap': 62, 'unknown': 64
    }
    tag = element.tag.split('}')[-1]
    tag_value = tag_mapping.get(tag, 32)

    element_id = element.attrib.get('id', '0')
    element_id_number = re.findall(r'\d+', element_id)
    if element_id_number:
        element_id_number = element_id_number[0]
    else:
        element_id_number = '0'
    tag_value = float(f"{tag_value}.{element_id_number:0>4}")

    opacity = float(element.attrib.get('opacity', 1.0))
    fill = element.attrib.get('fill', 'currentColor')
    stroke = element.attrib.get('stroke', 'currentColor')
    stroke_width = float(element.attrib.get('stroke-width', 1.0))

    if fill == 'currentColor':
        fill = get_inherited_attribute(element, 'fill') or 'black'
    if stroke == 'currentColor':
        stroke = get_inherited_attribute(element, 'stroke') or 'none'

    fill_h, fill_s, fill_l = get_color_features(fill, current_color)
    stroke_h, stroke_s, stroke_l = get_color_features(stroke, current_color)

    transform = element.attrib.get('transform', None)
    if transform:
        current_transform = f"{current_transform} {transform}"

    bbox_values = get_transformed_bbox(element, current_transform)

    layer = layer_extractor.get_node_layers().get(element_id, ['0'])
    layer = [int(part.split('_')[-1]) if part != '0' else 0 for part in layer]
    # print(layer)
    # layer = 1

    tag_name = element.attrib.get('tag_name', '')

    return [
        tag_name, tag_value, opacity, fill_h, fill_s, fill_l,
        stroke_h, stroke_s, stroke_l, stroke_width,
        layer, *bbox_values
    ]


def process_svg(file_path):
    svg_parser = SVGParser(file_path)
    root = svg_parser.run().getroot()

    layer_extractor = LayerDataExtractor()
    layer_extractor.extract_layers(root)

    features = []
    elements = list(root.iter())
    for element in tqdm(elements, total=len(elements), desc="Processing SVG Elements"):
        if element.tag.split('}')[-1] in {'circle', 'rect', 'line', 'polyline', 'polygon', 'path', 'text', 'ellipse',
                                          'image', 'use'}:
            feature = extract_features(element, layer_extractor)
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
    # Load the CSV file into a DataFrame
    df = pd.read_csv(input_csv_path)

    # Initialize the list to store the JSON data
    json_data = []

    # Iterate through each row in the DataFrame
    for index, row in df.iterrows():
        # Extract the 'id' from the 'tag_name' column
        element_id = row['tag_name']

        # Extract the last 20 columns as Fourier features
        fourier_features = row[-20:].tolist()

        # print(fourier_features[9])
        # Append the formatted data to the list
        json_data.append({
            "id": element_id,
            "fourier_features": fourier_features
        })

    # Write the result to a JSON file
    with open(output_json_path, 'w') as json_file:
        json.dump(json_data, json_file, indent=4)

def save_svg_with_ids(svg_input_path, svg_output_path):
    svgid(svg_input_path, svg_output_path)


def process_and_save_features(svg_input_path, output_csv_path, output_svg_with_ids_path):
    # 处理 SVG 文件并保存特征
    features = process_svg(svg_input_path)
    save_features(features, output_csv_path)

    # 保存带有 ID 的 SVG 文件
    save_svg_with_ids(svg_input_path, output_svg_with_ids_path)
