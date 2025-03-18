from lxml import etree
import re
import os
from tqdm import tqdm

class SVGModifier:
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir

    def modify_and_save_svgs(self):
        # 如果输出目录不存在，则创建它
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # 获取输入目录中的所有SVG文件
        files = os.listdir(self.input_dir)
        svg_files = [f for f in files if f.lower().endswith('.svg')]
        
        # 使用tqdm显示进度条
        for idx, file_name in enumerate(tqdm(svg_files, desc="Processing SVG files"), 1):
            input_path = os.path.join(self.input_dir, file_name)
            output_path = os.path.join(self.output_dir, f'{idx}.svg')
            try:
                self.modify_and_save_svg(input_path, output_path)
            except Exception as e:
                print(f"Error processing {input_path}: {e}")

    def modify_and_save_svg(self, input_path, output_path):
        self.load_svg(input_path)
        self.embed_css_styles()
        self.save_svg(output_path)

    def load_svg(self, input_path):
        # 加载SVG文件
        parser = etree.XMLParser(remove_blank_text=True)
        self.tree = etree.parse(input_path, parser)
        self.root = self.tree.getroot()
        self._remove_namespace()

    def _remove_namespace(self):
        # 删除SVG元素中的命名空间
        for elem in self.root.getiterator():
            if not hasattr(elem.tag, 'find'):
                continue
            i = elem.tag.find('}')
            if i >= 0:
                elem.tag = elem.tag[i + 1:]

    def embed_css_styles(self):
        # 将嵌入的CSS样式直接展开并嵌入到对应的SVG元素中
        style_element = self.root.find('.//style')
        if style_element is not None:
            css_text = style_element.text.strip()
            css_rules = re.findall(r'\.(st\d+)\s*\{([^\}]+)\}', css_text)
            for cls, style in css_rules:
                elements = self.root.xpath(f"//*[@class='{cls}']")
                style_dict = dict(re.findall(r'([\w-]+)\s*:\s*([^;]+);?', style))
                for elem in elements:
                    for prop, val in style_dict.items():
                        elem.attrib[prop] = val
                    if 'class' in elem.attrib:
                        del elem.attrib['class']
            self.root.remove(style_element)

    def save_svg(self, output_path):
        # 保存修改后的SVG文件
        self.tree.write(output_path, encoding='utf-8', xml_declaration=True, pretty_print=True)

# 使用新方法批量处理SVG文件
svg_modifier = SVGModifier('./public/newData6', './public/newData6')
svg_modifier.modify_and_save_svgs()
