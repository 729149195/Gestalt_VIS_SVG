# import json
# import colorsys
# import os
# import numpy as np
# from tqdm import tqdm

# class SVGFeatureExtractor:
#     def __init__(self, json_file_path, output_dir, num_buckets=15):
#         self.json_file_path = json_file_path
#         self.output_dir = output_dir
#         os.makedirs(self.output_dir, exist_ok=True)
#         self.num_buckets = num_buckets

#     def process_specific_json_file(self):
#         self.process_file(self.json_file_path)

#     def tag_mapping(self, svg_tag):
#         tag_map = {tag_name: key for key, tag_name in enumerate([
#             "svg", "g", "path", "rect", "circle", "ellipse", "line", "polyline",
#             "polygon", "text", "image", "use", "defs", "symbol", "marker", "pattern",
#             "mask", "filter", "linearGradient", "radialGradient", "stop", "clipPath",
#             "textPath", "tspan", "a", "foreignObject", "solidColor", "linearGradient", "radialGradient", "pattern",
#             "hatch", "mesh", "title"
#         ])}
#         return tag_map.get(svg_tag, -1)

#     def hex_to_rgb(self, hex_color):
#         hex_color = hex_color.lstrip('#')
#         lv = len(hex_color)
#         if lv == 3:
#             hex_color = ''.join([c * 2 for c in hex_color])
#         elif lv != 6:
#             return 0.0, 0.0, 0.0
#         if all(c in '0123456789abcdefABCDEF' for c in hex_color):
#             return tuple(int(hex_color[i:i + 2], 16) for i in range(0, 6, 2))
#         return 0.0, 0.0, 0.0

#     def rgb_to_hsl(self, rgb):
#         r_normalized, g_normalized, b_normalized = [x / 255.0 for x in rgb]
#         h, l, s = colorsys.rgb_to_hls(r_normalized, g_normalized, b_normalized)
#         return round(h * 360.0, 3), round(s * 100.0, 3), round(l * 100.0, 3)

#     def fill_mapping(self, fill_color):
#         if fill_color and fill_color != "none":
#             rgb = self.hex_to_rgb(fill_color)
#             hsl = self.rgb_to_hsl(rgb)
#             return hsl
#         return 0.0, 0.0, 0.0

#     def fill_layer(self, layer):
#         layer = layer[1:]
#         integer_part = ''.join([str(int(num) % 10) if len(num) > 2 else num for num in layer[:-1]])
#         final_str = f"{integer_part}.{layer[-1]}"
#         final_float = float(final_str)
#         return final_float

#     def extract_bbox_features(self, bbox_data):
#         if not bbox_data:
#             return [0.0] * 9
#         is_complex_shape = any(isinstance(subitem, (list, tuple)) for item in bbox_data for subitem in item)
#         if is_complex_shape:
#             valid_subitems = [subitem for item in bbox_data for subitem in item if
#                               isinstance(subitem, (list, tuple)) and len(subitem) == 2]
#             min_left = min(subitem[0] for subitem in valid_subitems) if valid_subitems else 0
#             min_top = min(subitem[1] for subitem in valid_subitems) if valid_subitems else 0
#             max_right = max(subitem[0] for subitem in valid_subitems) if valid_subitems else 0
#             max_bottom = max(subitem[1] for subitem in valid_subitems) if valid_subitems else 0
#         else:
#             min_left = min(item[0] for item in bbox_data) if bbox_data else 0
#             min_top = min(item[1] for item in bbox_data) if bbox_data else 0
#             max_right = max(item[0] for item in bbox_data) if bbox_data else 0
#             max_bottom = max(item[1] for item in bbox_data) if bbox_data else 0
#         center_x = round((min_left + max_right) / 2, 3)
#         center_y = round((min_top + max_bottom) / 2, 3)
#         width = round(max_right - min_left, 3)
#         height = round(max_bottom - min_top, 3)
#         area = round(width * height, 3)
#         return [round(min_top, 3), round(max_bottom, 3), round(min_left, 3), round(max_right, 3), center_x, center_y,
#                 width, height, area]

#     def process_file(self, json_file_path):
#         output_json_file = os.path.join(self.output_dir, os.path.basename(json_file_path).replace('.json', '.json'))
#         with open(json_file_path, 'r', encoding='utf-8') as f:
#             data = json.load(f)

#         output_json = {"feature": {}}
#         features = []

#         for node_id, node_data in tqdm(data.items(), desc=f"Processing {os.path.basename(json_file_path)}",
#                                        colour='green'):
#             feature_vector = []
#             tag = self.tag_mapping(node_data.get("tag", "").split('_')[0])
#             feature_vector.append(tag)

#             opacity_encoded = node_data.get("opacity", 1)
#             feature_vector.append(float(opacity_encoded))

#             fill_encoded = self.fill_mapping(node_data.get("fill", ""))
#             feature_vector.extend(fill_encoded)

#             stroke_encoded = self.fill_mapping(node_data.get("stroke", ""))
#             feature_vector.extend(stroke_encoded)

#             stroke_width_encoded = node_data.get("stroke-width", 1.0)
#             feature_vector.append(float(stroke_width_encoded))

#             stroke_dasharray_encoded = 0.0 if node_data.get("stroke-dasharray", "") == "" else 1.0
#             feature_vector.append(float(stroke_dasharray_encoded))

#             layer_encoded = self.fill_layer(node_data.get("layer", []))
#             feature_vector.append(layer_encoded)

#             bbox_encoded = self.extract_bbox_features(node_data.get("bbox", []))
#             feature_vector.extend(bbox_encoded)

#             output_json["feature"][node_id] = feature_vector
#             features.append(feature_vector)

#         # 转换为numpy数组以便于处理
#         features = np.array(features, dtype=float)
        
#         # 归一化处理，除了第一列和第二列外的所有列
#         for col in range(2, features.shape[1]):  # 从第二列开始
#             col_data = features[:, col]
#             min_val = np.min(col_data)
#             max_val = np.max(col_data)
#             # 防止除以0的情况
#             if max_val > min_val:
#                 features[:, col] = (col_data - min_val) / (max_val - min_val)

#         # 写入归一化后的特征向量到 JSON 文件
#         for i, node_id in enumerate(output_json["feature"].keys()):
#             output_json["feature"][node_id] = features[i].tolist()

#         with open(output_json_file, 'w', encoding='utf-8') as json_file:
#             json.dump(output_json, json_file, indent=4)

#     def process_all_json_files(self):
#         json_files = [os.path.join(self.input_file, f) for f in os.listdir(self.input_file) if f.endswith('.json')]
#         for json_file in tqdm(json_files, desc="Overall Progress", colour='blue'):
#             self.process_file(json_file)

# # # 示例使用
# # json_file_path = './GMoutput/extracted_nodes.json'
# # output_dir = './modules/Contrastive_Clustering/test'
# # extractor = SVGFeatureExtractor(json_file_path=json_file_path, output_dir=output_dir)
# # extractor.process_specific_json_file()


# 未归一化
import json
import colorsys
import os
import numpy as np
from tqdm import tqdm

class SVGFeatureExtractor:
    def __init__(self, json_file_path, output_dir, num_buckets=15):
        self.json_file_path = json_file_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.num_buckets = num_buckets

    def process_specific_json_file(self):
        self.process_file(self.json_file_path)

    def tag_mapping(self, svg_tag):
        tag_map = {tag_name: key for key, tag_name in enumerate([
            "svg", "g", "path", "rect", "circle", "ellipse", "line", "polyline",
            "polygon", "text", "image", "use", "defs", "symbol", "marker", "pattern",
            "mask", "filter", "linearGradient", "radialGradient", "stop", "clipPath",
            "textPath", "tspan", "a", "foreignObject", "solidColor", "linearGradient", "radialGradient", "pattern",
            "hatch", "mesh", "title"
        ])}
        return tag_map.get(svg_tag, -1)

    def hex_to_rgb(self, hex_color):
        hex_color = hex_color.lstrip('#')
        lv = len(hex_color)
        if lv == 3:
            hex_color = ''.join([c * 2 for c in hex_color])
        elif lv != 6:
            return 0.0, 0.0, 0.0
        if all(c in '0123456789abcdefABCDEF' for c in hex_color):
            return tuple(int(hex_color[i:i + 2], 16) for i in range(0, 6, 2))
        return 0.0, 0.0, 0.0

    def rgb_to_hsl(self, rgb):
        r_normalized, g_normalized, b_normalized = [x / 255.0 for x in rgb]
        h, l, s = colorsys.rgb_to_hls(r_normalized, g_normalized, b_normalized)
        return round(h * 360.0, 3), round(s * 100.0, 3), round(l * 100.0, 3)

    def fill_mapping(self, fill_color):
        if fill_color and fill_color != "none":
            rgb = self.hex_to_rgb(fill_color)
            hsl = self.rgb_to_hsl(rgb)
            return hsl
        return 0.0, 0.0, 0.0

    def fill_layer(self, layer):
        layer = layer[1:]
        integer_part = ''.join([str(int(num) % 10) if len(num) > 2 else num for num in layer[:-1]])
        final_str = f"{integer_part}.{layer[-1]}"
        final_float = float(final_str)
        return final_float

    def extract_bbox_features(self, bbox_data):
        if not bbox_data:
            return [0.0] * 9
        is_complex_shape = any(isinstance(subitem, (list, tuple)) for item in bbox_data for subitem in item)
        if is_complex_shape:
            valid_subitems = [subitem for item in bbox_data for subitem in item if
                              isinstance(subitem, (list, tuple)) and len(subitem) == 2]
            min_left = min(subitem[0] for subitem in valid_subitems) if valid_subitems else 0
            min_top = min(subitem[1] for subitem in valid_subitems) if valid_subitems else 0
            max_right = max(subitem[0] for subitem in valid_subitems) if valid_subitems else 0
            max_bottom = max(subitem[1] for subitem in valid_subitems) if valid_subitems else 0
        else:
            min_left = min(item[0] for item in bbox_data) if bbox_data else 0
            min_top = min(item[1] for item in bbox_data) if bbox_data else 0
            max_right = max(item[0] for item in bbox_data) if bbox_data else 0
            max_bottom = max(item[1] for item in bbox_data) if bbox_data else 0
        center_x = round((min_left + max_right) / 2, 3)
        center_y = round((min_top + max_bottom) / 2, 3)
        width = round(max_right - min_left, 3)
        height = round(max_bottom - min_top, 3)
        area = round(width * height, 3)
        return [round(min_top, 3), round(max_bottom, 3), round(min_left, 3), round(max_right, 3), center_x, center_y,
                width, height, area]

    def process_file(self, json_file_path):
        output_json_file = os.path.join(self.output_dir, os.path.basename(json_file_path).replace('.json', '.json'))
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        output_json = {"feature": {}}
        features = []

        for node_id, node_data in tqdm(data.items(), desc=f"Processing {os.path.basename(json_file_path)}",
                                       colour='green'):
            feature_vector = []
            tag = self.tag_mapping(node_data.get("tag", "").split('_')[0])
            feature_vector.append(tag)

            opacity_encoded = node_data.get("opacity", 1)
            feature_vector.append(float(opacity_encoded))

            fill_encoded = self.fill_mapping(node_data.get("fill", ""))
            feature_vector.extend(fill_encoded)

            stroke_encoded = self.fill_mapping(node_data.get("stroke", ""))
            feature_vector.extend(stroke_encoded)

            stroke_width_encoded = node_data.get("stroke-width", 1.0)
            feature_vector.append(float(stroke_width_encoded))

            stroke_dasharray_encoded = 0.0 if node_data.get("stroke-dasharray", "") == "" else 1.0
            feature_vector.append(float(stroke_dasharray_encoded))

            layer_encoded = self.fill_layer(node_data.get("layer", []))
            feature_vector.append(layer_encoded)

            bbox_encoded = self.extract_bbox_features(node_data.get("bbox", []))
            feature_vector.extend(bbox_encoded)

            output_json["feature"][node_id] = feature_vector
            features.append(feature_vector)

        # 转换为numpy数组以便于处理
        features = np.array(features, dtype=float)

        # 直接写入未归一化的特征向量到 JSON 文件
        for i, node_id in enumerate(output_json["feature"].keys()):
            output_json["feature"][node_id] = features[i].tolist()

        with open(output_json_file, 'w', encoding='utf-8') as json_file:
            json.dump(output_json, json_file, indent=4)

    def process_all_json_files(self):
        json_files = [os.path.join(self.input_file, f) for f in os.listdir(self.input_file) if f.endswith('.json')]
        for json_file in tqdm(json_files, desc="Overall Progress", colour='blue'):
            self.process_file(json_file)

# # 示例使用
# json_file_path = './GMoutput/extracted_nodes.json'
# output_dir = './modules/Contrastive_Clustering/test'
# extractor = SVGFeatureExtractor(json_file_path=json_file_path, output_dir=output_dir)
# extractor.process_specific_json_file()
