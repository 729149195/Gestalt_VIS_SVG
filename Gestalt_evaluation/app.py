from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from flask_socketio import SocketIO
import os
import json
from bs4 import BeautifulSoup
import traceback
import io
import matplotlib
import queue
import time

matplotlib.use('Agg')  # 在导入 pyplot 之前设置后端
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 'FangSong']
matplotlib.rcParams['axes.unicode_minus'] = False

# 导入自定义模块
from static.modules import featureCSV
from static.modules import normalized_features_liner_mds_2 as normalized_features
from static.modules.cluster import main as run_clustering
from static.modules.average_equivalent_mapping import EquivalentWeightsCalculator
from static.modules.subgraph_detection import main as run_subgraph_detection

# 创建一个全局的进度队列
progress_queue = queue.Queue()

# 初始化Flask应用
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# 配置常量
UPLOAD_FOLDER = 'static/uploadSvg'
DATA_FOLDER = 'static/data'

# 确保目录存在
for folder in [UPLOAD_FOLDER, DATA_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# 配置应用
app.config.update(
    UPLOAD_FOLDER=UPLOAD_FOLDER,
    DATA_FOLDER=DATA_FOLDER
)

# 工具函数
def get_file_path(filename, folder='DATA_FOLDER'):
    """获取文件完整路径"""
    base_folder = app.config[folder]
    return os.path.join(base_folder, filename)

# API路由
@app.route('/')
def hello_world():
    return 'Hello World!'

def send_progress_update(progress, step):
    """
    发送进度更新到队列
    """
    progress_queue.put({
        'progress': progress,
        'step': step
    })

@app.route('/progress')
def progress_stream():
    """
    Server-Sent Events 端点，用于发送进度更新
    """
    def generate():
        while True:
            try:
                # 非阻塞方式获取进度更新
                data = progress_queue.get_nowait()
                yield f"data: {json.dumps(data)}\n\n"
            except queue.Empty:
                # 如果队列为空，等待一小段时间
                time.sleep(0.1)
                continue

    return Response(generate(), mimetype='text/event-stream')

def process_svg_file(file_path):
    """处理上传的SVG文件的核心逻辑"""
    try:
        print(f"开始处理SVG文件: {file_path}")
        send_progress_update(5, "开始处理SVG文件...")
        
        # 设置输出路径
        output_paths = {
            'csv': get_file_path('features.csv'),
            'svg_with_ids': os.path.join(app.config['UPLOAD_FOLDER'], 'svg_with_ids.svg'),
            'init_json': get_file_path('init_json.json'),
            'normalized_init_json': get_file_path('normalized_init_json.json'),
            'community_data': get_file_path('community_data_mult.json'),
            'features_data': get_file_path('cluster_features.json'),
            'normalized_csv': get_file_path('normalized_features.csv')
        }
        
        # 文件检查
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"输入文件不存在: {file_path}")
        if os.path.getsize(file_path) == 0:
            raise ValueError(f"输入文件为空: {file_path}")

        # 处理步骤
        print("Start extracting features...")
        send_progress_update(10, "Feature extraction in progress...")
        featureCSV.process_and_save_features(file_path, output_paths['csv'], output_paths['svg_with_ids'])
        
        print("Starting to standardise features...")
        send_progress_update(25, "Being standardised features...")
        normalized_features.normalize_features(output_paths['csv'], output_paths['normalized_csv'])
        
        print("Data format being processed...")
        send_progress_update(35, "Data format being processed...")
        featureCSV.process_csv_to_json(output_paths['csv'], output_paths['init_json'])
        featureCSV.process_csv_to_json(output_paths['normalized_csv'], output_paths['normalized_init_json'])
        
        print("计算等价权重...")
        send_progress_update(60, "Equivalent weights being calculated...")
        calculator = EquivalentWeightsCalculator(model_path="static/modules/model_feature_dim_4_batch_64.tar")
        calculator.compute_and_save_equivalent_weights(
            output_paths['normalized_csv'],
            output_file_avg='static/data/average_equivalent_mapping.json',
            output_file_all='static/data/equivalent_weights_by_tag.json'
        )

        # 运行聚类
        print("开始运行对比学习模型，输出特征表示...")
        send_progress_update(75, "Comparative learning model being run...")
        run_clustering(
            output_paths['normalized_csv'],
            output_paths['community_data'],
            output_paths['features_data'],
        )

        # 运行子图检测
        print("开始运行子图检测...")
        send_progress_update(85, "Subgraph detection in progress...")
        run_subgraph_detection(
            features_json_path=output_paths['features_data'],
            output_dir=os.path.dirname(output_paths['features_data']),
            # louvain/gmm
            clustering_method='gmm',  
            subgraph_dimensions=[[0], [1], [2], [3], [0,1], [0,2], [0,3], [1,2], [1,3], [2,3],
                               [0,1,2], [0,1,3], [0,2,3], [1,2,3], [0,1,2,3]],
            progress_callback=send_progress_update
        )

        send_progress_update(100, "Processing completed")
        print("Processing completed")
        return {
            'success': True,
            'data': {
                'svg_file': file_path,
                **output_paths,
                'svg_with_ids_file': f"/{output_paths['svg_with_ids']}"
            }
        }
    except Exception as e:
        print(f"处理SVG文件时出错: {str(e)}")
        print(f"错误堆栈: {traceback.format_exc()}")
        send_progress_update(0, f"处理出错: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }

def process_svg_styles(svg_content):
    """
    处理SVG中的style标签,将样式直接作为属性应用到对应元素
    """
    soup = BeautifulSoup(svg_content, 'xml')
    
    # 处理style标签
    style_tag = soup.find('style')
    if style_tag:
        print("style标签，开始理样式...")
        # 解析CSS样式
        style_content = style_tag.string
        style_rules = {}
        if style_content:
            # 分割样式规则
            rules = style_content.split('}')
            for rule in rules:
                if '{' in rule:
                    selector, styles = rule.split('{')
                    selector = selector.strip()
                    styles = styles.strip()
                    if selector.startswith('.'):  # 处理类选择器
                        class_name = selector[1:]  # 移除点号
                        # 解析各个样式属性
                        style_attrs = {}
                        for style in styles.split(';'):
                            style = style.strip()
                            if ':' in style:
                                prop, value = style.split(':')
                                prop = prop.strip()
                                value = value.strip()
                                style_attrs[prop] = value
                        style_rules[class_name] = style_attrs
        
        # 应用样式到元素
        for class_name, style_attrs in style_rules.items():
            elements = soup.find_all(class_=class_name)
            for element in elements:
                # 直接将样式作为属性添加到元素
                for prop, value in style_attrs.items():
                    element[prop] = value
                    
        # 移除style标签
        style_tag.decompose()
    
    return str(soup)

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
            
            # 读取SVG内容
            svg_content = file.read().decode('utf-8')
            
            # 添加详细的日志
            print("SVG内容读取成功，长度:", len(svg_content))
            
            try:
                # 处理样式
                processed_svg = process_svg_styles(svg_content)
                print("样式处理成功")
            except Exception as style_error:
                print(f"样式处理失败: {str(style_error)}")
                return jsonify({'error': f'Error processing styles: {str(style_error)}'}), 500
            
            # 保存处理后的文件
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(processed_svg)
                
            print(f"文件已保存到: {file_path}")

            try:
                # 使用featureCSV处理SVG，添加ID
                temp_csv_path = os.path.join(app.config['DATA_FOLDER'], 'temp_features.csv')
                output_svg_with_ids_path = os.path.join(app.config['UPLOAD_FOLDER'], 'svg_with_ids.svg')
                
                # 添加详细的参数检查
                print(f"处理参数检查:")
                print(f"- 输入文件存在: {os.path.exists(file_path)}")
                print(f"- 输入文件大小: {os.path.getsize(file_path)}")
                
                featureCSV.process_and_save_features(file_path, temp_csv_path, output_svg_with_ids_path)
                print("SVG处理完成")
                
            except Exception as process_error:
                print(f"SVG处理失败: {str(process_error)}")
                print(f"错误类型: {type(process_error)}")
                print(f"堆栈跟踪: {traceback.format_exc()}")
                return jsonify({'error': f'Error processing SVG: {str(process_error)}'}), 500
            
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
            print(f"错误类型: {type(e)}")
            print(f"堆栈跟踪: {traceback.format_exc()}")
            return jsonify({'error': f'Error processing file: {str(e)}'}), 500
    else:
        return jsonify({'error': 'Invalid file type'}), 400

@app.route('/process', methods=['POST'])
def process_file():
    filename = request.json.get('filename')
    if not filename:
        return jsonify({'error': 'No filename provided'}), 400

    # 只有不是以 generated_ 开头的文件才添加 uploaded_ 前缀
    if not filename.startswith('generated_') and not filename.startswith('uploaded_'):
        filename = f'uploaded_{filename}'

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

# 获取生成 SVG 文件内容
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

    # 定义子图数文件的路径
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
    """
    try:
        # 首先处理样式
        svg_content = process_svg_styles(svg_content)
        
        soup = BeautifulSoup(svg_content, 'xml')
        svg = soup.find('svg')
        if not svg:
            print("未找到SVG根元素")
            return svg_content

        # 保存svg的原始属性
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
                
                # 递归处理每个子元素收集结果
                has_valid_children = False
                for child in children:
                    if child.name:  # 确保是元素节点
                        result = process_element(child)
                        if result:
                            element.append(result)
                            has_valid_children = True
                
                # 如果组内有有效子元素，保留该组
                return element if has_valid_children else None
            
            # 元素过滤逻辑
            if element.name in selected_elements:
                element_id = element.get('id')
                # 印调试信息
                print(f"检查元素: {element.name}, ID: {element_id}")
                
                # 如果有选中的ID列表且不为空
                if selected_ids and len(selected_ids) > 0:
                    # 保元素有ID且在选中列表中
                    if element_id and element_id in selected_ids:
                        print(f"保留元素: {element_id}")
                        return element
                    print(f"过滤掉元素: {element_id}")
                    return None
                else:
                    # 没有选中的ID列表，保留所有选中类型的元素
                    print(f"保留元素(按类型): {element.name}")
                    return element
            
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
        preserved_elements = []
        for element in svg.children:
            if element.name:  # 确保是元素节点
                processed = process_element(element)
                if processed:
                    preserved_elements.append(processed)
                    new_svg.append(processed)

        # 打印保留的元素数量
        print(f"保留的元素数量: {len(preserved_elements)}")
        
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
        print(f"错误堆栈: {traceback.format_exc()}")
        raise

@app.route('/filter_and_process', methods=['POST'])
def filter_and_process():
    try:
        data = request.json
        original_filename = data.get('filename')
        selected_elements = data.get('selectedElements', [])
        selected_node_ids = data.get('selectedNodeIds', [])  # 获取选中的节点ID
        
        print(f"开始处理文件: {original_filename}")
        print(f"选中的元素类型: {selected_elements}")
        print(f"选中的节点ID: {selected_node_ids}")
        
        if not original_filename or not selected_elements:
            return jsonify({
                'success': False,
                'error': 'Missing filename or selected elements'
            }), 400
        
        # 只有不是以 generated_ 开头的文件才添加 uploaded_ 前缀
        if not original_filename.startswith('generated_') and not original_filename.startswith('uploaded_'):
            original_filename = f'uploaded_{original_filename}'
        
        # 读取原始SVG文件
        original_file_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
        if not os.path.exists(original_file_path):
            print(f"原始文件不存在: {original_file_path}")
            return jsonify({
                'success': False,
                'error': 'Original file not found'
            }), 404
            
        # 首先处理原始文件生成带ID的SVG
        output_csv_path = os.path.join(app.config['DATA_FOLDER'], 'temp_features.csv')
        svg_with_ids_path = os.path.join(app.config['UPLOAD_FOLDER'], 'svg_with_ids.svg')
        
        print("生成带ID的SVG文件...")
        featureCSV.process_and_save_features(original_file_path, output_csv_path, svg_with_ids_path)
        
        # 删除临时CSV文件
        if os.path.exists(output_csv_path):
            os.remove(output_csv_path)
            
        # 读取带ID的SVG文件
        print(f"读取带ID的SVG文件: {svg_with_ids_path}")
        with open(svg_with_ids_path, 'r', encoding='utf-8') as f:
            svg_content = f.read()
        
        # 过滤SVG内容
        print("开始过滤SVG内容")
        filtered_svg = filter_svg_elements(svg_content, selected_elements, selected_node_ids)
        
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
            
        # 只有不是以 generated_ 开头的文件才添加 uploaded_ 前缀
        if not filename.startswith('generated_') and not filename.startswith('uploaded_'):
            filename = f'uploaded_{filename}'
            
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
            'polyline', 'polygon', 'path', 'text', 'image' ,'title'
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

# 添加新的路由来获取网格识别结果
@app.route('/grid_structures/<int:dimension>', methods=['GET'])
def get_grid_structures(dimension):
    """
    获取指定维度的网格结构检测结果
    """
    grid_file_path = os.path.join(app.config['DATA_FOLDER'], 
                                 f'grid_structures/grid_detection_dimension_{dimension}.json')
    
    if os.path.exists(grid_file_path):
        with open(grid_file_path, 'r', encoding='utf-8') as f:
            return jsonify({
                'success': True,
                'data': json.load(f)
            }), 200
    else:
        return jsonify({
            'success': False,
            'error': f'Grid structure file not found for dimension {dimension}'
        }), 404

@app.route('/api/matplotlib', methods=['POST'])
def handle_matplotlib():
    try:
        # 获取请求数据
        data = request.get_json()
        if not data or 'code' not in data:
            return jsonify({'error': 'Missing code in request'}), 400

        # 创建一个内存缓冲区来保存SVG
        buffer = io.StringIO()

        try:
            # 清理之前的图形
            plt.close('all')
            
            # 重置matplotlib的样式设置
            plt.style.use('default')
            
            # 重新设置字体配置
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 'FangSong']
            plt.rcParams['axes.unicode_minus'] = False

            # 创建新的图形
            plt.figure(figsize=(10, 6))

            # 执行代码
            exec(data['code'], {'plt': plt, 'np': np})
            
            # 获取当前图形
            fig = plt.gcf()
            
            # 将图形保存为SVG格式到缓冲区
            fig.savefig(buffer, format='svg', bbox_inches='tight', dpi=300)
            
            # 获取SVG内容
            svg_content = buffer.getvalue()
            
            # 清理资源
            plt.close(fig)
            buffer.close()
            
            return svg_content, 200, {'Content-Type': 'image/svg+xml'}
            
        except Exception as e:
            plt.close('all')  # 确保清理所有图形
            raise e
            
    except Exception as e:
        print(f"Matplotlib错误: {str(e)}")
        print(f"错误堆栈: {traceback.format_exc()}")
        return jsonify({'error': f'Matplotlib error: {str(e)}'}), 500
    finally:
        try:
            buffer.close()
        except:
            pass
        plt.close('all')  # 确保清理所有图形

@app.route('/clear_upload_folder', methods=['POST'])
def clear_upload_folder():
    """清空上传文件夹的接口"""
    try:
        # 获取上传文件夹路径
        upload_folder = app.config['UPLOAD_FOLDER']
        
        # 确保文件夹存在
        if not os.path.exists(upload_folder):
            return jsonify({
                'success': True,
                'message': 'Upload folder does not exist'
            }), 200
            
        # 删除文件夹中的所有文件
        for filename in os.listdir(upload_folder):
            file_path = os.path.join(upload_folder, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                    print(f"已删除文件: {filename}")
            except Exception as e:
                print(f"删除文件 {filename} 时出错: {str(e)}")
                
        return jsonify({
            'success': True,
            'message': 'Upload folder cleared successfully'
        }), 200
        
    except Exception as e:
        print(f"清空上传文件夹时出错: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/calculate_gmm', methods=['POST'])
def calculate_gmm():
    """计算GMM结果的API端点"""
    try:
        data = request.get_json()
        if not data or 'values' not in data:
            return jsonify({'error': 'Missing values in request'}), 400

        values = np.array(data['values'])
        
        # 数据验证
        if len(values) < 2:
            return jsonify({
                'success': False,
                'error': '数据点数量不足，至少需要2个数据点'
            }), 400

        # 检查数据是否都相同
        if np.all(values == values[0]):
            return jsonify({
                'success': True,
                'components': [{
                    'mean': float(values[0]),
                    'variance': 0.0,
                    'weight': 1.0
                }],
                'bic_scores': [0.0]
            }), 200

        # 标准化数据
        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(values.reshape(-1, 1)).ravel()
        
        # 根据数据量动态调整最大聚类数
        max_components = min(5, len(values) - 1)
        if max_components < 2:
            max_components = 2
        
        n_components_range = range(2, max_components + 1)
        bic = []
        gmm_models = []
        
        for n_components in n_components_range:
            # 增加迭代次数和收敛阈值
            gmm = GaussianMixture(
                n_components=n_components,
                random_state=42,
                max_iter=200,
                tol=1e-4,
                reg_covar=1e-6  # 增加协方差矩阵的正则化
            )
            gmm.fit(scaled_values.reshape(-1, 1))
            bic.append(gmm.bic(scaled_values.reshape(-1, 1)))
            gmm_models.append(gmm)
        
        # 选择BIC最小的模型
        best_idx = np.argmin(bic)
        best_n_components = n_components_range[best_idx]
        best_gmm = gmm_models[best_idx]
        
        # 获取每个组件的参数
        components = []
        for i in range(best_n_components):
            # 将标准化的参数转换回原始尺度
            mean = scaler.inverse_transform([[best_gmm.means_[i][0]]])[0][0]
            variance = best_gmm.covariances_[i][0][0] * (scaler.scale_[0] ** 2)
            
            components.append({
                'mean': float(mean),
                'variance': float(variance),
                'weight': float(best_gmm.weights_[i])
            })
        
        # 按均值排序组件
        components.sort(key=lambda x: x['mean'])
        
        return jsonify({
            'success': True,
            'components': components,
            'bic_scores': [float(score) for score in bic]
        }), 200
        
    except Exception as e:
        print(f"GMM计算错误: {str(e)}")
        print(f"错误堆栈: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': f'GMM计算错误: {str(e)}'
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
