from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO
import os
from static.modules import featureCSV as featureCSV
from static.modules import normalized_features_liner as normalized_features
from static.modules.cluster import main as run_clustering
from static.modules.cluster import get_eps
from static.modules.draw_graph import draw_element_nodes_with_lines
from static.modules.average_equivalent_mapping import EquivalentWeightsCalculator
from bs4 import BeautifulSoup
import copy

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")  # 允许跨域

# 确保上传目录存在
UPLOAD_FOLDER = 'static/uploadSvg'
DATA_FOLDER = 'static/data'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DATA_FOLDER'] = DATA_FOLDER

max_eps = None
epss = None
@app.route('/')
def hello_world():
    return 'Hello World!'

def process_svg_file(file_path):
    """
    处理上传的SVG文件的核心逻辑
    """
    try:
        print(f"开始处理SVG文件: {file_path}")
        # 设置输出路径
        output_csv_path = os.path.join(app.config['DATA_FOLDER'], 'features.csv')
        output_svg_with_ids_path = os.path.join(app.config['UPLOAD_FOLDER'], 'svg_with_ids.svg')
        temp_svg_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_processed.svg')
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"输入文件不存在: {file_path}")

        # 检查文件是否为空
        if os.path.getsize(file_path) == 0:
            raise ValueError(f"输入文件为空: {file_path}")

        print("开始提取特征...")
        # 根据是否是过滤后的文件决定输出路径
        if 'filtered_' in os.path.basename(file_path):
            # 处理过滤后的文件时使用临时文件
            featureCSV.process_and_save_features(file_path, output_csv_path, temp_svg_path)
            # 处理完成后删除临时文件
            if os.path.exists(temp_svg_path):
                os.remove(temp_svg_path)
        else:
            # 只有处理原始文件时才更新 svg_with_ids.svg
            featureCSV.process_and_save_features(file_path, output_csv_path, output_svg_with_ids_path)
        
        init_json = os.path.join(app.config['DATA_FOLDER'], 'init_json.json')
        normalized_init_json = os.path.join(app.config['DATA_FOLDER'], 'normalized_init_json.json')
        community_data_path = os.path.join(app.config['DATA_FOLDER'], 'community_data_mult.json')
        features_data_path = os.path.join(app.config['DATA_FOLDER'], 'cluster_features.json')
        normalized_csv_path = os.path.join(app.config['DATA_FOLDER'], 'normalized_features.csv')

        global max_eps, epss

        print("开始标准化特征...")
        normalized_features.normalize_features(output_csv_path, normalized_csv_path)

        print("处理CSV到JSON...")
        featureCSV.process_csv_to_json(output_csv_path, init_json)
        featureCSV.process_csv_to_json(normalized_csv_path, normalized_init_json)

        print("计算等价权重...")
        calculator = EquivalentWeightsCalculator(model_path="static/modules/checkpoint_newData_ZT_200.32.007.tar")
        calculator.compute_and_save_equivalent_weights(normalized_csv_path, 
            output_file_avg='static/data/average_equivalent_mapping.json', 
            output_file_all='static/data/equivalent_weights_by_tag.json')

        print("获取聚类参数...")
        epss, max_eps = get_eps(normalized_csv_path, community_data_path, features_data_path)
        
        if not isinstance(max_eps, (int, float)) or max_eps <= 0:
            print(f"警告: max_eps 值无效 ({max_eps})，使用默认值0.4")
            max_eps = 0.4

        print(f"运行聚类算法，max_eps={max_eps}...")
        run_clustering(normalized_csv_path, community_data_path, features_data_path, max_eps)

        print("处理完成")
        return {
            'success': True,
            'data': {
                'svg_file': file_path,
                'csv_file': output_csv_path,
                'normalized_csv_file': normalized_csv_path,
                'community_data_mult': community_data_path,
                'cluster_features': features_data_path,
                'svg_with_ids_file': f'/{output_svg_with_ids_path}'
            }
        }
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"处理SVG文件时出错: {str(e)}")
        print(f"错误堆栈: {error_trace}")
        return {
            'success': False,
            'error': str(e)
        }

# 文件上传的路由
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and file.filename.endswith('.svg'):
        try:
            print(f"开始处理上传文件: {file.filename}")
            # 保存上传的文件
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            print(f"文件已保存到: {file_path}")

            # 立即处理SVG，添加ID
            output_svg_with_ids_path = os.path.join(app.config['UPLOAD_FOLDER'], 'svg_with_ids.svg')
            
            # 使用featureCSV模块处理SVG，添加ID
            temp_csv_path = os.path.join(app.config['DATA_FOLDER'], 'temp_features.csv')
            featureCSV.process_and_save_features(file_path, temp_csv_path, output_svg_with_ids_path)
            
            # 删除临时CSV文件
            if os.path.exists(temp_csv_path):
                os.remove(temp_csv_path)

            return jsonify({
                'success': True,
                'message': 'File uploaded and processed successfully',
                'filename': file.filename
            }), 200
        except Exception as e:
            print(f"上传处理出错: {str(e)}")
            return jsonify({'error': f'Error processing file: {str(e)}'}), 500
    else:
        return jsonify({'error': 'Invalid file type'}), 400

@app.route('/process', methods=['POST'])
def process_file():
    filename = request.json.get('filename')
    if not filename:
        return jsonify({'error': 'No filename provided'}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404

    try:
        # 处理文件
        result = process_svg_file(file_path)
        if result['success']:
            return jsonify({
                'success': 'File processed successfully',
                **result['data']
            }), 200
        else:
            return jsonify({
                'error': f'Error processing file: {result["error"]}'
            }), 500
    except Exception as e:
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500

@app.route('/run_clustering', methods=['POST'])
def run_clustering_with_params():
    eps = request.json.get('eps', 0.4)
    min_samples = request.json.get('min_samples', 1)
    distance_threshold_ratio = request.json.get('distance_threshold_ratio', 0.4)

    try:
        eps = float(eps)
        min_samples = int(min_samples)
        distance_threshold_ratio = float(distance_threshold_ratio)

        if eps <= 0:
            raise ValueError("eps must be greater than 0")

        if min_samples <= 0:
            raise ValueError("min_samples must be greater than 0")

        if distance_threshold_ratio <= 0 or distance_threshold_ratio >= 1:
            raise ValueError("distance_threshold_ratio must be between 0 and 1")

        normalized_csv_path = os.path.join(app.config['DATA_FOLDER'], 'normalized_features.csv')
        community_data_path = os.path.join(app.config['DATA_FOLDER'], 'community_data_mult.json')
        features_data_path = os.path.join(app.config['DATA_FOLDER'], 'cluster_features.json')

        run_clustering(normalized_csv_path, community_data_path, features_data_path, eps, min_samples, distance_threshold_ratio)

        return jsonify({
            'success': 'Clustering executed with specified parameters',
            'eps': eps,
            'min_samples': min_samples,
            'distance_threshold_ratio':distance_threshold_ratio,
            'normalized_csv_file': normalized_csv_path,
            'community_data_mult': community_data_path,
            'cluster_features': features_data_path
        }), 200
    except Exception as e:
        return jsonify({'error': 'An error occurred while running clustering', 'details': str(e)}), 500


@app.route('/get_eps_list', methods=['GET'])
def get_eps_list():
    global max_eps, epss  # 声明使用全局变量
    # print(max_eps,epss)
    return jsonify({"max_eps":max_eps, "epss":epss}),200

# 获取生成的 SVG 文件内容
@app.route('/get_svg', methods=['GET'])
def get_svg():
    try:
        # 修改为从uploadSvg目录读取文件
        svg_file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'svg_with_ids.svg')
        print(f"尝试读取SVG文件: {svg_file_path}")

        if not os.path.exists(svg_file_path):
            print("SVG文件不存在")
            return jsonify({'error': 'SVG file not found'}), 404

        with open(svg_file_path, 'r', encoding='utf-8') as svg_file:
            svg_content = svg_file.read()
            
            # 确保SVG内容有效
            if not svg_content.strip().startswith('<?xml') and not svg_content.strip().startswith('<svg'):
                svg_content = f'<?xml version="1.0" encoding="UTF-8"?>\n{svg_content}'
            
            # 确保SVG有正确的命名空间
            if 'xmlns=' not in svg_content:
                svg_content = svg_content.replace('<svg', '<svg xmlns="http://www.w3.org/2000/svg"')
                
        print("SVG文件读取成功")
        return svg_content, 200, {
            'Content-Type': 'image/svg+xml; charset=utf-8',
            'Cache-Control': 'no-cache'
        }
    except Exception as e:
        print(f"获取SVG出错: {str(e)}")
        return jsonify({'error': f'Error reading SVG: {str(e)}'}), 500


@app.route('/community_data_mult', methods=['GET'])
def get_community_data_mult():
    community_data_path = os.path.join(app.config['DATA_FOLDER'], 'community_data_mult.json')

    if os.path.exists(community_data_path):
        with open(community_data_path, 'r', encoding='utf-8') as json_file:
            return json_file.read(), 200, {'Content-Type': 'application/json'}
    else:
        return jsonify({'error': 'community_data_mult.json file not found'}), 404

@app.route('/init_json', methods=['GET'])
def get_init_json():
    community_data_path = os.path.join(app.config['DATA_FOLDER'], 'init_json.json')

    if os.path.exists(community_data_path):
        with open(community_data_path, 'r', encoding='utf-8') as json_file:
            return json_file.read(), 200, {'Content-Type': 'application/json'}
    else:
        return jsonify({'error': 'init_json.json file not found'}), 404

@app.route('/normalized_init_json', methods=['GET'])
def get_normalized_init_json():
    community_data_path = os.path.join(app.config['DATA_FOLDER'], 'normalized_init_json.json')

    if os.path.exists(community_data_path):
        with open(community_data_path, 'r', encoding='utf-8') as json_file:
            return json_file.read(), 200, {'Content-Type': 'application/json'}
    else:
        return jsonify({'error': 'normalized_init_json.json file not found'}), 404


@app.route('/attr_num_data', methods=['GET'])
def histogram_attr_data():
    community_data_path = os.path.join(app.config['DATA_FOLDER'], 'attr_num.json')

    if os.path.exists(community_data_path):
        with open(community_data_path, 'r', encoding='utf-8') as json_file:
            return json_file.read(), 200, {'Content-Type': 'application/json'}
    else:
        return jsonify({'error': 'community_data_mult.json file not found'}), 404


@app.route('/ele_num_data', methods=['GET'])
def histogram_ele_data():
    community_data_path = os.path.join(app.config['DATA_FOLDER'], 'ele_num.json')

    if os.path.exists(community_data_path):
        with open(community_data_path, 'r', encoding='utf-8') as json_file:
            return json_file.read(), 200, {'Content-Type': 'application/json'}
    else:
        return jsonify({'error': 'community_data_mult.json file not found'}), 404


@app.route('/bbox_num_data', methods=['GET'])
def histogram_bbox_data():
    community_data_path = os.path.join(app.config['DATA_FOLDER'], 'bbox_points_count.json')

    if os.path.exists(community_data_path):
        with open(community_data_path, 'r', encoding='utf-8') as json_file:
            return json_file.read(), 200, {'Content-Type': 'application/json'}
    else:
        return jsonify({'error': 'community_data_mult.json file not found'}), 404


@app.route('/bottom_position', methods=['GET'])
def bottom_position():
    community_data_path = os.path.join(app.config['DATA_FOLDER'], 'Bottom_data.json')

    if os.path.exists(community_data_path):
        with open(community_data_path, 'r', encoding='utf-8') as json_file:
            return json_file.read(), 200, {'Content-Type': 'application/json'}
    else:
        return jsonify({'error': 'community_data_mult.json file not found'}), 404


@app.route('/right_position', methods=['GET'])
def right_data():
    community_data_path = os.path.join(app.config['DATA_FOLDER'], 'Right_data.json')

    if os.path.exists(community_data_path):
        with open(community_data_path, 'r', encoding='utf-8') as json_file:
            return json_file.read(), 200, {'Content-Type': 'application/json'}
    else:
        return jsonify({'error': 'community_data_mult.json file not found'}), 404


@app.route('/cluster_features', methods=['GET'])
def cluster_features():
    community_data_path = os.path.join(app.config['DATA_FOLDER'], 'cluster_features.json')

    if os.path.exists(community_data_path):
        with open(community_data_path, 'r', encoding='utf-8') as json_file:
            return json_file.read(), 200, {'Content-Type': 'application/json'}
    else:
        return jsonify({'error': 'community_data_mult.json file not found'}), 404


@app.route('/fill_num', methods=['GET'])
def fill_data():
    community_data_path = os.path.join(app.config['DATA_FOLDER'], 'fill_num.json')

    if os.path.exists(community_data_path):
        with open(community_data_path, 'r', encoding='utf-8') as json_file:
            return json_file.read(), 200, {'Content-Type': 'application/json'}
    else:
        return jsonify({'error': 'community_data_mult.json file not found'}), 404


@app.route('/group_data', methods=['GET'])
def histogram_group_data():
    community_data_path = os.path.join(app.config['DATA_FOLDER'], 'group_data.json')

    if os.path.exists(community_data_path):
        with open(community_data_path, 'r', encoding='utf-8') as json_file:
            return json_file.read(), 200, {'Content-Type': 'application/json'}
    else:
        return jsonify({'error': 'community_data_mult.json file not found'}), 404


@app.route('/layer_data', methods=['GET'])
def data_layer():
    community_data_path = os.path.join(app.config['DATA_FOLDER'], 'layer_data.json')

    if os.path.exists(community_data_path):
        with open(community_data_path, 'r', encoding='utf-8') as json_file:
            return json_file.read(), 200, {'Content-Type': 'application/json'}
    else:
        return jsonify({'error': 'community_data_mult.json file not found'}), 404


@app.route('/left_position', methods=['GET'])
def left_position():
    community_data_path = os.path.join(app.config['DATA_FOLDER'], 'Left_data.json')

    if os.path.exists(community_data_path):
        with open(community_data_path, 'r', encoding='utf-8') as json_file:
            return json_file.read(), 200, {'Content-Type': 'application/json'}
    else:
        return jsonify({'error': 'community_data_mult.json file not found'}), 404


@app.route('/stroke_num', methods=['GET'])
def stroke_data():
    community_data_path = os.path.join(app.config['DATA_FOLDER'], 'stroke_num.json')

    if os.path.exists(community_data_path):
        with open(community_data_path, 'r', encoding='utf-8') as json_file:
            return json_file.read(), 200, {'Content-Type': 'application/json'}
    else:
        return jsonify({'error': 'community_data_mult.json file not found'}), 404



@app.route('/average_equivalent_mapping', methods=['GET'])
def average_equivalent_mapping():
    community_data_path = os.path.join(app.config['DATA_FOLDER'], 'average_equivalent_mapping.json')

    if os.path.exists(community_data_path):
        with open(community_data_path, 'r', encoding='utf-8') as json_file:
            return json_file.read(), 200, {'Content-Type': 'application/json'}
    else:
        return jsonify({'error': 'average_equivalent_mapping.json file not found'}), 404


@app.route('/equivalent_weights_by_tag', methods=['GET'])
def equivalent_weights_by_tag():
    community_data_path = os.path.join(app.config['DATA_FOLDER'], 'equivalent_weights_by_tag.json')

    if os.path.exists(community_data_path):
        with open(community_data_path, 'r', encoding='utf-8') as json_file:
            return json_file.read(), 200, {'Content-Type': 'application/json'}
    else:
        return jsonify({'error': 'equivalent_weights_by_tag.json file not found'}), 404


@app.route('/top_position', methods=['GET'])
def top_position():
    community_data_path = os.path.join(app.config['DATA_FOLDER'], 'Top_data.json')

    if os.path.exists(community_data_path):
        with open(community_data_path, 'r', encoding='utf-8') as json_file:
            return json_file.read(), 200, {'Content-Type': 'application/json'}
    else:
        return jsonify({'error': 'community_data_mult.json file not found'}), 404


@app.route('/subgraph/<int:dimension>', methods=['GET'])
def get_subgraph_data(dimension):
    # 确保 dimension 参数在有效围内（假设有4个特征维度，即0到3）
    if dimension < 0 or dimension > 3:
        return jsonify({'error': 'Invalid dimension'}), 400

    # 定义子图数据文件的路径
    subgraph_file_path = os.path.join(app.config['DATA_FOLDER'], f'subgraphs/subgraph_dimension_{dimension}.json')

    # 检查文件是否存在
    if os.path.exists(subgraph_file_path):
        with open(subgraph_file_path, 'r', encoding='utf-8') as json_file:
            return json_file.read(), 200, {'Content-Type': 'application/json'}
    else:
        return jsonify({'error': f'subgraph_dimension_{dimension}.json file not found'}), 404

def filter_svg_elements(svg_content, selected_elements, selected_ids=None):
    """
    过滤SVG文件，支持按元素类型和具体元素ID过滤
    
    Args:
        svg_content: SVG文件内容
        selected_elements: 选中的元素类型列表
        selected_ids: 选中的具体元素ID列表，如果为None则只按类型过滤
    """
    try:
        soup = BeautifulSoup(svg_content, 'xml')
        svg = soup.find('svg')
        if not svg:
            print("未找到SVG根元素")
            return svg_content

        # 保存svg的��始属性
        svg_attrs = dict(svg.attrs)
        
        # 递归处理元素
        def process_element(element):
            # 如果是组元素，递归处理其子元素
            if element.name == 'g':
                # 记录原始子元素
                children = list(element.children)
                # 清空当前组
                element.clear()
                # 恢复原始属性
                for attr, value in element.attrs.items():
                    element[attr] = value
                
                # 递归处理每个子元素
                for child in children:
                    if child.name:  # 确保是元素节点
                        result = process_element(child)
                        if result:
                            element.append(result)
                
                # 如果组内还有子元素，保留该组
                return element if len(element.contents) > 0 else None
            
            # 如果是选中的元素类型，且（没有指定ID过滤或元素ID在选中列表中），则保留它
            elif (element.name in selected_elements and 
                  (selected_ids is None or element.get('id') in selected_ids)):
                return element
            
            # 其他元素不保留
            return None

        # 创建新的svg元素并保留原始属性
        new_svg = soup.new_tag('svg')
        for attr, value in svg_attrs.items():
            new_svg[attr] = value
            
        # 添加必要的命名空间
        if 'xmlns' not in svg_attrs:
            new_svg['xmlns'] = "http://www.w3.org/2000/svg"
        if 'xmlns:xlink' not in svg_attrs:
            new_svg['xmlns:xlink'] = "http://www.w3.org/1999/xlink"

        # 处理所有顶层元素
        for element in svg.children:
            if element.name:  # 确保是元素节点
                processed = process_element(element)
                if processed:
                    new_svg.append(processed)

        # 替换原始的svg元素
        svg.replace_with(new_svg)
        
        print(f"过滤完成，保留的元素类型: {selected_elements}")
        if selected_ids:
            print(f"保留的元素ID: {selected_ids}")
            
        result = str(soup)
        
        # 确保XML声明和编码正确
        if not result.strip().startswith('<?xml'):
            result = '<?xml version="1.0" encoding="UTF-8"?>\n' + result
            
        return result
        
    except Exception as e:
        print(f"过滤SVG元素时出错: {str(e)}")
        raise

@app.route('/filter_and_process', methods=['POST'])
def filter_and_process():
    try:
        data = request.json
        original_filename = data.get('filename')
        selected_elements = data.get('selectedElements', [])
        
        print(f"开始处理文件: {original_filename}")
        print(f"选中的元素类型: {selected_elements}")
        
        if not original_filename or not selected_elements:
            return jsonify({
                'success': False,
                'error': 'Missing filename or selected elements'
            }), 400
        
        # 读取原始SVG文件
        original_file_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
        if not os.path.exists(original_file_path):
            print(f"原始文件不存在: {original_file_path}")
            return jsonify({
                'success': False,
                'error': 'Original file not found'
            }), 404
            
        print(f"读取原始文件: {original_file_path}")
        with open(original_file_path, 'r', encoding='utf-8') as f:
            svg_content = f.read()
        
        # 过滤SVG内容
        print("开始过滤SVG内容")
        filtered_svg = filter_svg_elements(svg_content, selected_elements)
        
        # 保存过滤后的SVG文件
        filtered_filename = f'filtered_{original_filename}'
        filtered_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filtered_filename)
        print(f"保存过滤后的文件: {filtered_file_path}")
        with open(filtered_file_path, 'w', encoding='utf-8') as f:
            f.write(filtered_svg)
        
        # 处理过滤后的文件
        print("开始处理过滤后的文件")
        result = process_svg_file(filtered_file_path)
        
        if result['success']:
            print("文件处理成功")
            return jsonify({
                'success': True,
                'message': 'File filtered and processed successfully',
                **result['data']
            }), 200
        else:
            print(f"文件处理失败: {result.get('error', 'Unknown error')}")
            return jsonify({
                'success': False,
                'error': result.get('error', 'Unknown error')
            }), 500
            
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"处理过程出错: {str(e)}")
        print(f"错误堆栈: {error_trace}")
        return jsonify({
            'success': False,
            'error': str(e),
            'trace': error_trace
        }), 500

@app.route('/get_visible_elements', methods=['POST'])
def get_visible_elements():
    try:
        data = request.json
        filename = data.get('filename')
        print(f"开始获取可见元素，文件名: {filename}")
        
        if not filename:
            return jsonify({'error': 'No filename provided'}), 400
            
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print(f"查找文件路径: {file_path}")
        
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            return jsonify({'error': 'File not found'}), 404
            
        with open(file_path, 'r', encoding='utf-8') as f:
            svg_content = f.read()
            
        soup = BeautifulSoup(svg_content, 'xml')
        visible_elements = []
        
        # 定义可见元素类型
        visible_element_types = [
            'rect', 'circle', 'ellipse', 'line', 
            'polyline', 'polygon', 'path', 'text', 'image'
        ]
        
        # 获取所有SVG内部的元素
        svg = soup.find('svg')
        if not svg:
            print("未找到SVG根元素")
            return jsonify({'error': 'No SVG element found'}), 400
            
        # 统计每种类型的元素数量
        element_counts = {}
        for element_type in visible_element_types:
            elements = svg.find_all(element_type, recursive=True)
            count = len(elements)
            if count > 0:
                element_counts[element_type] = count
                visible_elements.append({
                    'id': element_type,
                    'tag': element_type,
                    'count': count
                })
        
        print("找到的可见元素类型及数量:")
        for element_type, count in element_counts.items():
            print(f"{element_type}: {count}")
        
        return jsonify({
            'success': True,
            'elements': visible_elements
        }), 200
        
    except Exception as e:
        print(f"获取可见元素出错: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# 添加新的路由处理元素ID过滤
@app.route('/filter_by_ids', methods=['POST'])
def filter_by_ids():
    try:
        data = request.json
        filename = data.get('filename')
        selected_elements = data.get('selectedElements', [])
        selected_ids = data.get('selectedIds', [])
        
        if not filename:
            return jsonify({'error': 'No filename provided'}), 400
            
        # 使用原始文件进行过滤
        original_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(original_file_path):
            return jsonify({'error': 'File not found'}), 404
            
        # 读取原始SVG文件
        with open(original_file_path, 'r', encoding='utf-8') as f:
            svg_content = f.read()
            
        # 过滤SVG内容
        filtered_svg = filter_svg_elements(svg_content, selected_elements, selected_ids)
        
        # 保存过滤后的SVG文件
        filtered_filename = f'filtered_{filename}'
        filtered_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filtered_filename)
        with open(filtered_file_path, 'w', encoding='utf-8') as f:
            f.write(filtered_svg)
            
        # 处理过滤后的文件
        result = process_svg_file(filtered_file_path)
        if not result['success']:
            raise Exception(result.get('error', 'Unknown error'))
            
        return jsonify({
            'success': True,
            'message': 'SVG filtered and processed successfully',
            'filtered_file': filtered_filename
        }), 200
        
    except Exception as e:
        print(f"处理过程出错: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
