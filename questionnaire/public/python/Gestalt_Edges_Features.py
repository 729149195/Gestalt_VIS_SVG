import json

# 假定的文件路径，这里作为示例，实际路径可能需要调整
input_nodes_file = './public/python/data/GMinfo.json'
output_nodes_file = './public/python/data/extracted_nodes.json'


class NodeExtractor:
    @staticmethod
    def parse_points_from_string(points_str):
        points_list = []
        points_pairs = points_str.split()
        for pair in points_pairs:
            x_str, y_str = pair.split(',')
            points_list.append([float(x_str), float(y_str)])
        return points_list

    @staticmethod
    def calculate_bboxs_for_line_polygon_polyline(points):
        bboxs = []
        for i in range(len(points) - 1):
            start = points[i]
            end = points[i + 1]
            mid = [(start[0] + end[0]) / 2, (start[1] + end[1]) / 2]
            bbox = [start, end, mid]
            bboxs.append(bbox)
        return bboxs

    @staticmethod
    def extract_nodes_info(input_file, output_file):
        with open(input_file, 'r', encoding='utf-8') as file:
            data = json.load(file)

        nodes = data['DiGraph']['Nodes']
        extracted_nodes = {}

        for node, attributes in nodes.items():
            # 调试输出
            # print(f"Processing node: {node}")

            tag = attributes['Attributes']['tag']
            attrs = attributes['Attributes']['attributes']
            visible = attributes['Attributes']['visible']

            fill = attrs.get('fill', 'empty')
            stroke = attrs.get('stroke', 'empty')

            if not visible or (fill == stroke and fill == 'empty') or (fill == stroke and fill == 'empty'):
                # 调试输出
                # print(f"Skipping node: {node} (not visible or no fill/stroke)")
                continue

            level = attributes['Attributes']['level']
            layer = attributes['Attributes']['layer'].split('_')
            text_content = attributes['Attributes']['text_content']
            bbox = attributes['Attributes']['attributes']['bbox']

            extracted_attrs = {
                'tag': tag,
                'stroke': attrs.get('stroke', None),
                'stroke-width': attrs.get('stroke-width', 1),
                'stroke-opacity': attrs.get('stroke-opacity', 1),
                'fill': attrs.get('fill', None),
                'opacity': attrs.get('opacity', 1),
                'level': level,
                'layer': layer,
                'text_content': text_content,
            }

            if tag.split('_')[0] == "path":
                pcode = attrs.get('Pcode', [])
                pnums = attrs.get('Pnums', [])
                path_to_lines = PathToLines(pcode, pnums)
                bboxs = path_to_lines.get_bboxs()
                if len(bboxs) == 0:
                    bboxs = [[0, 0], [0, 0], [0, 0]]
                extracted_attrs['bbox'] = bboxs
                extracted_nodes[node] = extracted_attrs

            elif tag.split('_')[0] in ["polygon", "polyline"]:
                points_str = attrs.get('points', "")
                points = NodeExtractor.parse_points_from_string(points_str)
                bboxs = NodeExtractor.calculate_bboxs_for_line_polygon_polyline(points)
                extracted_attrs['bbox'] = bboxs
                extracted_nodes[node] = extracted_attrs

            elif visible:
                extracted_attrs['bbox'] = bbox
                extracted_nodes[node] = extracted_attrs

        with open(output_file, 'w', encoding='utf-8') as file:
            json.dump(extracted_nodes, file, indent=4)


class PathToLines:
    def __init__(self, pcode, pnums):
        self.last_control = None
        self.pcode = pcode
        self.pnums = pnums
        self.lines = []  # 保存线段
        self.bboxs = []  # 保存线段的bbox
        self.current_position = (0, 0)  # 初始化当前位置
        self.start_position = None  # 保存路径的起始位置
        self.last_control_point = None  # 保存最后一个控制点
        self.last_command = None  # 保存最后执行的命令

    def process_commands(self):
        for code, nums in zip(self.pcode, self.pnums):
            if code == 'M':
                self.move_to(nums)
            elif code == 'L':
                self.line_to(nums)
            elif code == 'H':
                self.horizontal_line_to(nums)
            elif code == 'V':
                self.vertical_line_to(nums)
            elif code == 'C':
                self.cubic_bezier_to(nums)
            elif code == 'S':
                self.smooth_cubic_bezier_to(nums)
            elif code == 'Q':
                self.quadratic_bezier_to(nums)
            elif code == 'T':
                self.smooth_quadratic_bezier_to(nums)
            elif code == 'Z':
                self.close_path()

    def move_to(self, nums):
        x, y = float(nums[0]), float(nums[1])
        self.current_position = (x, y)
        # 当执行M命令时，更新起始位置
        self.start_position = (x, y)
        self.last_command = 'M'

    def line_to(self, nums):
        x, y = float(nums[0]), float(nums[1])
        self.lines.append([self.current_position, (x, y)])
        self.calculate_line_bbox(self.current_position, (x, y))
        self.current_position = (x, y)

    def quadratic_bezier_to(self, nums):
        cx, cy, x, y = map(float, nums)
        start = self.current_position
        control = (cx, cy)
        end = (x, y)
        points = [start]
        for t in range(1, 11):
            t /= 10  # 将t从0变化到1
            bx = round((1 - t) ** 2 * start[0] + 2 * (1 - t) * t * control[0] + t ** 2 * end[0], 2)
            by = round((1 - t) ** 2 * start[1] + 2 * (1 - t) * t * control[1] + t ** 2 * end[1], 2)
            points.append((bx, by))
        points.append(end)

        # 对每个连续的点对绘制直线
        for i in range(len(points) - 1):
            self.lines.append([points[i], points[i + 1]])
            self.calculate_line_bbox(points[i], points[i + 1])

    def smooth_quadratic_bezier_to(self, nums):
        # 假设前一个贝塞尔曲线的控制点存储在 last_control 变量中
        # 如果没有前一个贝塞尔曲线（即这是第一个T命令），可以简化处理为直线
        if not hasattr(self, 'last_control') or self.last_control is None:
            self.line_to(nums)
        else:
            # 计算当前控制点为上一个控制点的反射点
            reflect_x = 2 * self.current_position[0] - self.last_control[0]
            reflect_y = 2 * self.current_position[1] - self.last_control[1]
            self.last_control = (reflect_x, reflect_y)  # 更新控制点
            x, y = float(nums[0]), float(nums[1])
            # 使用反射的控制点绘制二次贝塞尔曲线
            self.quadratic_bezier_to([reflect_x, reflect_y, x, y])

    def calculate_line_bbox(self, start, end):
        # 应用四舍五入，这里假设保留两位小数
        min_x, max_x = round(min(start[0], end[0]), 2), round(max(start[0], end[0]), 2)
        min_y, max_y = round(min(start[1], end[1]), 2), round(max(start[1], end[1]), 2)
        mid_x = round((start[0] + end[0]) / 2, 2)
        mid_y = round((start[1] + end[1]) / 2, 2)
        self.bboxs.append([[min_x, min_y], [max_x, max_y], [mid_x, mid_y]])

    def horizontal_line_to(self, nums):
        x = float(nums[0])
        self.lines.append([self.current_position, (x, self.current_position[1])])
        self.calculate_line_bbox(self.current_position, (x, self.current_position[1]))
        self.current_position = (x, self.current_position[1])

    def vertical_line_to(self, nums):
        y = float(nums[0])
        self.lines.append([self.current_position, (self.current_position[0], y)])
        self.calculate_line_bbox(self.current_position, (self.current_position[0], y))
        self.current_position = (self.current_position[0], y)

    def cubic_bezier_to(self, nums):
        c1x, c1y, c2x, c2y, x, y = map(float, nums)
        start = self.current_position
        control1 = (c1x, c1y)
        control2 = (c2x, c2y)
        end = (x, y)
        points = [start]
        steps = 10  # 曲线分割成多少段，可以根据需要调整

        for step in range(1, steps + 1):
            t = step / steps
            # 三次贝塞尔曲线方程
            bx = (1 - t) ** 3 * start[0] + 3 * (1 - t) ** 2 * t * control1[0] + 3 * (1 - t) * t ** 2 * control2[
                0] + t ** 3 * end[0]
            by = (1 - t) ** 3 * start[1] + 3 * (1 - t) ** 2 * t * control1[1] + 3 * (1 - t) * t ** 2 * control2[
                1] + t ** 3 * end[1]
            points.append((round(bx, 2), round(by, 2)))

        # 使用计算出的点更新线段和边界框列表
        for i in range(len(points) - 1):
            self.lines.append([points[i], points[i + 1]])
            self.calculate_line_bbox(points[i], points[i + 1])

        # 更新当前位置为曲线的结束点
        self.current_position = end

    def smooth_cubic_bezier_to(self, nums):
        # 新控制点和结束点
        c2x, c2y, x, y = map(float, nums)

        if self.last_command in ['C', 'S']:
            # 前一个命令是C或S，计算反射的控制点
            last_cx, last_cy = self.last_control_point
            # 计算反射点，即当前点关于最后一个控制点的对称点
            c1x = 2 * self.current_position[0] - last_cx
            c1y = 2 * self.current_position[1] - last_cy
        else:
            # 前一个命令不是C或S，第一个控制点与当前点相同
            c1x, c1y = self.current_position

        # 使用计算出的控制点和结束点绘制三次贝塞尔曲线
        self.cubic_bezier_to([c1x, c1y, c2x, c2y, x, y])

        # 更新最后一个控制点和命令
        self.last_control_point = (c2x, c2y)
        self.last_command = 'S'

    def close_path(self):
        # 如果存在起始位置，则将当前位置连接回起始位置
        if self.start_position:
            self.lines.append([self.current_position, self.start_position])
            self.calculate_line_bbox(self.current_position, self.start_position)
            # 更新当前位置为起始位置，闭合路径
            self.current_position = self.start_position
        self.last_command = 'Z'

    def get_bboxs(self):
        self.process_commands()
        return self.bboxs


def extract_nodes():
    extractor = NodeExtractor()
    extractor.extract_nodes_info(input_nodes_file, output_nodes_file)
