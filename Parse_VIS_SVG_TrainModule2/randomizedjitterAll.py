import os
import random
from xml.etree import ElementTree as ET
import colorsys
from bs4 import BeautifulSoup
import re

# 将十六进制颜色转换为RGB
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 6:
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    elif len(hex_color) == 3:
        r, g, b = int(hex_color[0]*2, 16), int(hex_color[1]*2, 16), int(hex_color[2]*2, 16)
    else:
        raise ValueError(f"Invalid hex color: {hex_color}")
    return r, g, b

# 将RGB(x, x, x)格式颜色转换为RGB元组
def rgb_to_tuple(rgb_color):
    numbers = re.findall(r'\d+', rgb_color)
    return tuple(map(int, numbers))

# 将RGB转换为HSL
def rgb_to_hsl(r, g, b):
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    return colorsys.rgb_to_hls(r, g, b)

# 随机抖动颜色并转换为HSL格式的函数
def randomize_color_to_hsl(color, hue_amount=0.05, lightness_amount=0.1, saturation_amount=0.1):
    try:
        if color.startswith('#'):  # 如果是十六进制颜色
            rgb_color = hex_to_rgb(color)
            h, l, s = rgb_to_hsl(*rgb_color)  # 转换为HSL
        elif color.startswith('rgb'):  # 如果是rgb()格式颜色
            rgb_color = rgb_to_tuple(color)
            h, l, s = rgb_to_hsl(*rgb_color)  # 转换为HSL
        else:
            return color  # 如果不是十六进制或RGB格式，直接返回原颜色

    except ValueError:
        return color  # 如果解析失败，返回原颜色

    # 只在原数值的10%范围内进行随机抖动
    h = (h + random.uniform(-hue_amount * h, hue_amount * h)) % 1.0
    l = max(0, min(1, l + random.uniform(-lightness_amount * l, lightness_amount * l)))
    s = max(0, min(1, s + random.uniform(-saturation_amount * s, saturation_amount * s)))

    # 返回新的HSL颜色格式
    return f'hsl({int(h * 360)}, {int(s * 100)}%, {int(l * 100)}%)'

# 递归地去除命名空间前缀
def strip_namespace(element):
    for el in element.iter():
        if '}' in el.tag:
            el.tag = el.tag.split('}', 1)[1]  # 去掉命名空间URI部分
    return element

# 处理单个SVG文件
def process_single_svg(file_path, output_dir, versions=10):
    # 使用BeautifulSoup解析并修复SVG格式问题
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            soup = BeautifulSoup(content, 'xml')  # 使用BeautifulSoup处理SVG
            svg_str = str(soup)
            tree = ET.ElementTree(ET.fromstring(svg_str))  # 将BeautifulSoup处理后的内容转换为ElementTree
            root = tree.getroot()
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return  # 跳过错误文件

    # 去除命名空间前缀
    root = strip_namespace(root)

    # 定义处理单个版本的函数
    def process_version(root, version_name, apply_rotation=False, rotation_angle=0):
        # 创建root元素的深拷贝
        new_root = ET.ElementTree(ET.fromstring(ET.tostring(root))).getroot()

        # 获取所有样式定义
        style_element = new_root.find('.//style')
        if style_element is None:
            print(f"No style element found in {file_path}")
            return new_root  # 返回原始root

        style_text = style_element.text.strip()

        # 使用正则表达式解析样式表，处理多行样式块
        pattern = re.compile(r'([^{]+)\{([^}]+)\}', re.MULTILINE | re.DOTALL)
        matches = pattern.findall(style_text)

        new_styles = []
        for selector, properties_block in matches:
            selector = selector.strip()
            properties_block = properties_block.strip()
            properties = re.findall(r'[^;]+;', properties_block)
            updated_properties = []
            for prop in properties:
                prop = prop.strip().rstrip(';')
                if ':' in prop:
                    prop_name, prop_value = prop.split(':', 1)
                    prop_name = prop_name.strip()
                    prop_value = prop_value.strip()
                    if prop_name in ['fill', 'stroke']:
                        # 将颜色转换为HSL并随机抖动
                        new_color = randomize_color_to_hsl(prop_value)
                        updated_properties.append(f'{prop_name}: {new_color}')
                    else:
                        updated_properties.append(f'{prop_name}: {prop_value}')
            # 拼接成新的样式
            properties_str = ';\n    '.join(updated_properties)
            new_style = f'{selector} {{\n    {properties_str};\n}}'
            new_styles.append(new_style)

        # 修改样式表文本
        style_element.text = '\n\n'.join(new_styles)

        # 如果需要旋转，应用旋转变换
        if apply_rotation:
            # 获取SVG的宽度和高度
            width_attr = new_root.get('width')
            height_attr = new_root.get('height')

            try:
                width = float(re.findall(r'[\d.]+', width_attr)[0]) if width_attr else None
                height = float(re.findall(r'[\d.]+', height_attr)[0]) if height_attr else None
            except Exception as e:
                print(f"Error parsing width/height in {file_path}: {e}")
                width = None
                height = None

            if width is None or height is None:
                # 如果无法获取宽度和高度，尝试从viewBox中获取
                viewBox = new_root.get('viewBox')
                if viewBox:
                    try:
                        viewBox_values = list(map(float, viewBox.strip().split()))
                        width = viewBox_values[2]
                        height = viewBox_values[3]
                    except Exception as e:
                        print(f"Error parsing viewBox in {file_path}: {e}")
                        return new_root
                else:
                    print(f"No width/height or viewBox found in {file_path}")
                    return new_root

            # 创建一个新的<g>元素
            g_element = ET.Element('g')

            # 将new_root的所有子元素移动到g_element中
            for child in list(new_root):
                new_root.remove(child)
                g_element.append(child)

            # 将g_element添加回new_root
            new_root.append(g_element)

            # 计算旋转中心点
            cx = width / 2
            cy = height / 2

            # 应用旋转变换
            transform = f"rotate({rotation_angle} {cx} {cy})"
            g_element.set('transform', transform)

        # 重新添加命名空间声明
        new_root.set("xmlns", "http://www.w3.org/2000/svg")

        return new_root

    # 处理颜色随机化的版本
    for version in range(1, versions + 1):
        version_name = f"_{version}"
        new_root = process_version(root, version_name)

        # 保存新的SVG文件
        base_name = os.path.basename(file_path)
        name, ext = os.path.splitext(base_name)
        new_file_name = os.path.join(output_dir, f"{name}{version_name}{ext}")
        new_tree = ET.ElementTree(new_root)
        new_tree.write(new_file_name, encoding='utf-8', xml_declaration=True)

    # 处理旋转的版本
    rotation_directions = [('clockwise', 90), ('counterclockwise', -90)]
    for direction, angle in rotation_directions:
        for version in range(1, 6):  # 每个方向生成5个版本
            version_name = f"_{direction}_{version}"
            new_root = process_version(root, version_name, apply_rotation=True, rotation_angle=angle)

            # 保存新的SVG文件
            base_name = os.path.basename(file_path)
            name, ext = os.path.splitext(base_name)
            new_file_name = os.path.join(output_dir, f"{name}{version_name}{ext}")
            new_tree = ET.ElementTree(new_root)
            new_tree.write(new_file_name, encoding='utf-8', xml_declaration=True)

# 批量处理目录下的所有SVG文件
def process_svg_folder(folder_path, output_dir, versions=10):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith('.svg'):
            file_path = os.path.join(folder_path, file_name)
            process_single_svg(file_path, output_dir, versions)

# 使用示例
input_folder = './svg_withoutaxis'  # 替换为你的SVG文件夹路径
output_folder = './randomsvg_withoutaxis'  # 输出文件夹
process_svg_folder(input_folder, output_folder)
