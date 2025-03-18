import os
import random
from xml.etree import ElementTree as ET
import webcolors
import colorsys

# 随机抖动颜色并转换为HSL格式的函数
def randomize_color_to_hsl(color, hue_amount=0.1, lightness_amount=0.1, saturation_amount=0.1):
    try:
        rgb_color = webcolors.name_to_rgb(color)
    except ValueError:
        try:
            rgb_color = webcolors.hex_to_rgb(color)
        except ValueError:
            try:
                rgb_color = webcolors.rgb_percent_to_rgb(color)
            except ValueError:
                return color  # 返回原始颜色，如果解析失败
    
    # 将RGB颜色转换为HLS（色相-亮度-饱和度）格式
    r, g, b = rgb_color.red / 255.0, rgb_color.green / 255.0, rgb_color.blue / 255.0
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    
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
def process_single_svg(file_path, output_dir, versions=5):
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    # 去除命名空间前缀
    root = strip_namespace(root)
    
    # 获取所有样式定义
    style_element = root.find('.//style')
    if style_element is None:
        print(f"No style element found in {file_path}")
        return
    
    styles = style_element.text.strip().split('\n')
    
    # 处理指定的样式
    for version in range(1, versions + 1):
        new_styles = []
        for style in styles:
            if '{' in style and 'fill:' in style:  # 检查style格式
                parts = style.split('{', 1)
                style_name = parts[0]
                properties = parts[1].rstrip('}').split(';')
                for i, prop in enumerate(properties):
                    if 'fill:' in prop:
                        color = prop.split(':')[1].strip()
                        new_color = randomize_color_to_hsl(color)
                        properties[i] = f'fill: {new_color}'
                new_style = f'{style_name} {{{";".join(properties).strip()};}}'
                new_styles.append(new_style)
            else:
                new_styles.append(style)  # 保留不处理的样式
        
        # 修改样式表
        style_element.text = '\n'.join(new_styles)

        
        # 重新添加命名空间声明
        root.set("xmlns", "http://www.w3.org/2000/svg")
        
        # 保存新的SVG文件
        base_name = os.path.basename(file_path)
        name, ext = os.path.splitext(base_name)
        new_file_name = os.path.join(output_dir, f"{name}_{version}{ext}")
        tree.write(new_file_name)

# 批量处理目录下的所有SVG文件
def process_svg_folder(folder_path, output_dir, versions=1):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith('.svg'):
            file_path = os.path.join(folder_path, file_name)
            process_single_svg(file_path, output_dir, versions)
            
# 使用示例
input_folder = './public/newData2/'  # 替换为你的SVG文件夹路径
output_folder = './public/randomsvg/'  # 输出文件夹
process_svg_folder(input_folder, output_folder)
