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
import math
import tempfile
import shutil

# from Gestalt_API.static.modules.batch_evaluation import BatchEvaluator
matplotlib.use('Agg')  # 在导入 pyplot 之前设置后端
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 'FangSong']
matplotlib.rcParams['axes.unicode_minus'] = False

# 导入自定义模块
from static.modules import featureCSV
from static.modules import normalized_features_liner_mds_2 as normalized_features
from static.modules.cluster import main as run_clustering
from static.modules.average_equivalent_mapping import EquivalentWeightsCalculator
from static.modules.subgraph_detection import main as run_subgraph_detection
from static.modules import posandprop

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
        # 不添加新ID和tag_name属性，只提取特征
        featureCSV.process_and_save_features(file_path, output_paths['csv'], output_paths['svg_with_ids'], 
                                           add_ids=False, add_tag_names=False)
        
        print("Starting to standardise features...")
        send_progress_update(25, "Being standardised features...")
        normalized_features.normalize_features(output_paths['csv'], output_paths['normalized_csv'])
        
        print("Data format being processed...")
        send_progress_update(35, "Data format being processed...")
        featureCSV.process_csv_to_json(output_paths['csv'], output_paths['init_json'])
        featureCSV.process_csv_to_json(output_paths['normalized_csv'], output_paths['normalized_init_json'])
        
        # 处理位置和属性信息
        print("Location and property information being processed...")
        send_progress_update(45, "Location and property information being processed...")
        posandprop.process_position_and_properties(
            output_paths['init_json'],
            file_path,
            app.config['DATA_FOLDER']
        )
        
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
                # 使用featureCSV处理SVG，添加ID和tag_name属性
                temp_csv_path = os.path.join(app.config['DATA_FOLDER'], 'temp_features.csv')
                output_svg_with_ids_path = os.path.join(app.config['UPLOAD_FOLDER'], 'svg_with_ids.svg')
                
                # 添加详细的参数检查
                print(f"处理参数检查:")
                print(f"- 输入文件存在: {os.path.exists(file_path)}")
                print(f"- 输入文件大小: {os.path.getsize(file_path)}")
                
                # 上传时需要添加ID和tag_name属性
                featureCSV.process_and_save_features(file_path, temp_csv_path, output_svg_with_ids_path, 
                                                   add_ids=True, add_tag_names=True)
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
    
# 获取生成 SVG 文件内容
@app.route('/get_upload_svg', methods=['GET'])
def get_upload_svg():
    try:
        # 尝试按优先级查找文件
        possible_filenames = [
            'generated_with_id.svg',    # 正确拼写版本
            'generated_width_id.svg',   # 原始拼写版本
            'svg_with_ids.svg'          # 如果前两个不存在，使用get_svg使用的文件
        ]
        
        svg_file_path = None
        found_file = None
        
        # 遍历所有可能的文件名
        for filename in possible_filenames:
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            print(f"尝试读取SVG文件: {temp_path}")
            if os.path.exists(temp_path):
                svg_file_path = temp_path
                found_file = filename
                print(f"找到SVG文件: {found_file}")
                
                # 读取找到的文件
                with open(svg_file_path, 'r', encoding='utf-8') as svg_file:
                    svg_content = svg_file.read()
                    
                    # 检查文件内容是否为有效的SVG
                    if '<parsererror' in svg_content or svg_content.strip().startswith('<html>'):
                        print(f"文件内容包含错误，不是有效的SVG: {found_file}")
                        # 继续尝试下一个文件
                        continue
                    
                    # 确保SVG内容有效
                    if not svg_content.strip().startswith('<?xml') and not svg_content.strip().startswith('<svg'):
                        svg_content = f'<?xml version="1.0" encoding="UTF-8"?>\n{svg_content}'
                    
                    # 确保SVG有正确的命名空间
                    if 'xmlns=' not in svg_content:
                        svg_content = svg_content.replace('<svg', '<svg xmlns="http://www.w3.org/2000/svg"')
                    
                    print(f"SVG文件读取成功: {found_file}")
                    return svg_content, 200, {
                        'Content-Type': 'image/svg+xml; charset=utf-8',
                        'Cache-Control': 'no-cache'
                    }
                
        # 如果所有文件都不存在或都无效
        print("所有可能的SVG文件都不存在或无效")
        
        # 检查是否能通过get_svg的逻辑生成文件
        svg_with_ids_path = os.path.join(app.config['UPLOAD_FOLDER'], 'svg_with_ids.svg')
        
        # 如果没有找到文件，但是存在上传的SVG文件，尝试生成
        uploaded_files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) 
                          if f.endswith('.svg') and (f.startswith('uploaded_') or not any(f.startswith(p) for p in ['generated_', 'svg_with_ids', 'filtered_']))]
        
        if uploaded_files:
            print(f"发现上传的文件，尝试处理: {uploaded_files[0]}")
            original_file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_files[0])
            
            # 尝试生成带ID的SVG文件
            try:
                from static.modules import featureCSV
                output_csv_path = os.path.join(app.config['DATA_FOLDER'], 'temp_features.csv')
                
                # 处理文件，生成带ID的SVG
                featureCSV.process_and_save_features(
                    original_file_path, output_csv_path, svg_with_ids_path, 
                    add_ids=True, add_tag_names=True
                )
                
                # 删除临时CSV文件
                if os.path.exists(output_csv_path):
                    os.remove(output_csv_path)
                    
                # 复制一份作为generated_with_id.svg
                generated_path = os.path.join(app.config['UPLOAD_FOLDER'], 'generated_with_id.svg')
                shutil.copy2(svg_with_ids_path, generated_path)
                
                # 读取生成的文件
                with open(generated_path, 'r', encoding='utf-8') as svg_file:
                    svg_content = svg_file.read()
                    
                    # 确保SVG内容有效
                    if not svg_content.strip().startswith('<?xml') and not svg_content.strip().startswith('<svg'):
                        svg_content = f'<?xml version="1.0" encoding="UTF-8"?>\n{svg_content}'
                    
                    # 确保SVG有正确的命名空间
                    if 'xmlns=' not in svg_content:
                        svg_content = svg_content.replace('<svg', '<svg xmlns="http://www.w3.org/2000/svg"')
                    
                    print(f"成功生成并读取SVG文件: generated_with_id.svg")
                    return svg_content, 200, {
                        'Content-Type': 'image/svg+xml; charset=utf-8',
                        'Cache-Control': 'no-cache'
                    }
                
            except Exception as gen_error:
                print(f"自动生成SVG文件失败: {str(gen_error)}")
        
        # 如果没有找到或无法生成有效的SVG文件
        print("无法找到或生成有效的SVG文件")
        return jsonify({
            'error': 'SVG file not found or invalid. Try calling /get_svg endpoint first.',
            'suggestion': '请先调用/get_svg接口以生成必要的文件'
        }), 404
            
    except Exception as e:
        print(f"获取SVG出错: {str(e)}")
        print(f"错误堆栈: {traceback.format_exc()}")
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

@app.route('/width_position', methods=['GET'])
def width_position():
    width_data_path = os.path.join(app.config['DATA_FOLDER'], 'Width_data.json')

    if os.path.exists(width_data_path):
        with open(width_data_path, 'r', encoding='utf-8') as json_file:
            return json_file.read(), 200, {'Content-Type': 'application/json'}
    else:
        return jsonify({'error': 'Width_data.json file not found'}), 404

@app.route('/height_position', methods=['GET'])
def height_position():
    height_data_path = os.path.join(app.config['DATA_FOLDER'], 'Height_data.json')

    if os.path.exists(height_data_path):
        with open(height_data_path, 'r', encoding='utf-8') as json_file:
            return json_file.read(), 200, {'Content-Type': 'application/json'}
    else:
        return jsonify({'error': 'Height_data.json file not found'}), 404
    
@app.route('/element_colors', methods=['GET'])
def element_colors():
    try:
        csv_path = os.path.join(app.config['DATA_FOLDER'], 'features.csv')
        
        if not os.path.exists(csv_path):
            return jsonify({'error': 'features.csv file not found'}), 404
            
        # 读取CSV文件并提取颜色信息
        color_data = {}
        import csv
        
        with open(csv_path, 'r', encoding='utf-8') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                # 获取原始tag_name
                original_tag_name = row['tag_name']
                
                # 只保留最后一个"/"后的所有字符
                tag_name = original_tag_name.split('/')[-1]
                
                # 提取HSL颜色值
                fill_h = float(row['fill_h'])
                fill_s = float(row['fill_s'])
                fill_l = float(row['fill_l'])
                
                # 检查是否有有效的颜色值（有些元素可能没有填充颜色）
                if fill_h >= 0 and fill_s >= 0 and fill_l >= 0:
                    # 格式化为HSL字符串
                    hsl_color = f"hsl({round(fill_h, 1)}, {round(fill_s, 1)}%, {round(fill_l, 1)}%)"
                    color_data[tag_name] = hsl_color
                else:
                    # 如果没有有效的填充颜色，使用默认值或标记为无颜色
                    color_data[tag_name] = "none"
        
        return jsonify(color_data), 200
        
    except Exception as e:
        print(f"获取元素颜色信息时出错: {str(e)}")
        print(f"错误堆栈: {traceback.format_exc()}")
        return jsonify({'error': f'Error getting element colors: {str(e)}'}), 500


@app.route('/element_stroke_colors', methods=['GET'])
def element_stroke_colors():
    try:
        csv_path = os.path.join(app.config['DATA_FOLDER'], 'features.csv')
        
        if not os.path.exists(csv_path):
            return jsonify({'error': 'features.csv file not found'}), 404
            
        # 读取CSV文件并提取颜色信息
        color_data = {}
        import csv
        
        with open(csv_path, 'r', encoding='utf-8') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                # 获取原始tag_name
                original_tag_name = row['tag_name']
                
                # 只保留最后一个"/"后的所有字符
                tag_name = original_tag_name.split('/')[-1]
                
                # 提取HSL颜色值
                stroke_h = float(row['stroke_h'])
                stroke_s = float(row['stroke_s'])
                stroke_l = float(row['stroke_l'])
                
                # 检查是否有有效的颜色值（有些元素可能没有填充颜色）
                if stroke_h >= 0 and stroke_s >= 0 and stroke_l >= 0:
                    # 格式化为HSL字符串
                    hsl_color = f"hsl({round(stroke_h, 1)}, {round(stroke_s, 1)}%, {round(stroke_l, 1)}%)"
                    color_data[tag_name] = hsl_color
                else:
                    # 如果没有有效的填充颜色，使用默认值或标记为无颜色
                    color_data[tag_name] = "none"
        
        return jsonify(color_data), 200
        
    except Exception as e:
        print(f"获取元素颜色信息时出错: {str(e)}")
        print(f"错误堆栈: {traceback.format_exc()}")
        return jsonify({'error': f'Error getting element colors: {str(e)}'}), 500


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
            
        # 查找svg_with_ids.svg文件，这个文件应该是上传时生成的带有最初ID和tag_name的SVG
        svg_with_ids_path = os.path.join(app.config['UPLOAD_FOLDER'], 'svg_with_ids.svg')
        
        # 确保svg_with_ids.svg文件存在
        if not os.path.exists(svg_with_ids_path):
            print(f"带ID的SVG文件不存在: {svg_with_ids_path}")
            
            # 尝试使用generated_with_id.svg 
            svg_with_ids_path = os.path.join(app.config['UPLOAD_FOLDER'], 'generated_with_id.svg')
            if not os.path.exists(svg_with_ids_path):
                return jsonify({
                    'success': False,
                    'error': 'Could not find SVG with IDs. Please upload the file first.'
                }), 404
                
        print(f"使用带ID的SVG文件: {svg_with_ids_path}")
        
        # 首先处理原始文件生成带ID的SVG
        output_csv_path = os.path.join(app.config['DATA_FOLDER'], 'temp_features.csv')
        
        print("处理SVG文件以提取特征...")
        # 不添加新ID和tag_name属性，使用已有的
        featureCSV.process_and_save_features(original_file_path, output_csv_path, svg_with_ids_path, 
                                           add_ids=False, add_tag_names=False)
        
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

@app.route('/original_features', methods=['GET'])
def original_features():
    """获取原始的features.csv文件内容"""
    community_data_path = os.path.join(app.config['DATA_FOLDER'], 'features.csv')

    if os.path.exists(community_data_path):
        with open(community_data_path, 'r', encoding='utf-8') as csv_file:
            return csv_file.read(), 200, {'Content-Type': 'text/csv'}
    else:
        return jsonify({'error': 'features.csv file not found'}), 404

@app.route('/modify_and_calculate_salience', methods=['POST'])
def modify_and_calculate_salience():
    """
    修改SVG元素的属性并计算显著性
    
    请求参数:
    - modify_elements: 需要修改的元素列表，每个元素包含ids和要修改的属性
        [{
            "ids": ["元素ID1", "元素ID2", ...],  # 元素ID数组
            "attributes": {"属性名": "属性值"}   # 应用于所有指定元素的属性
        }]
    - debug: 布尔值，是否输出详细调试信息
    
    返回:
    - success: 布尔值，表示操作是否成功
    - salience: 显著性值
    - debug_info: 如果debug=True，返回详细的调试信息
    """
    try:
        data = request.json
        modify_elements = data.get('modify_elements', [])
        debug_mode = data.get('debug', False)
        
        # 用于存储调试信息
        debug_info = {}
        
        # 检查必要的参数
        if not modify_elements:
            return jsonify({'success': False, 'error': 'No elements to modify'}), 400
        
        # 获取与/get_svg相同的SVG文件
        svg_with_ids_path = os.path.join(app.config['UPLOAD_FOLDER'], 'svg_with_ids.svg')
        if not os.path.exists(svg_with_ids_path):
            # 如果svg_with_ids.svg不存在，尝试使用generated_with_id.svg
            svg_with_ids_path = os.path.join(app.config['UPLOAD_FOLDER'], 'generated_with_id.svg')
            if not os.path.exists(svg_with_ids_path):
                return jsonify({'success': False, 'error': 'No SVG file found. Please upload the file first.'}), 404
        
        if debug_mode:
            debug_info['svg_file_path'] = svg_with_ids_path
            
        # 读取SVG文件内容
        with open(svg_with_ids_path, 'r', encoding='utf-8') as f:
            svg_content = f.read()
        
        # 解析SVG内容，提取所有元素ID作为感知范围
        try:
            soup = BeautifulSoup(svg_content, 'xml')
            
            # 提取所有具有id属性的元素，排除id="svg"的元素
            scope_elements = []
            elements_with_id = soup.find_all(id=True)
            for element in elements_with_id:
                element_id = element.get('id')
                if element_id and element_id != 'svg':
                    scope_elements.append(element_id)
            
            if not scope_elements:
                return jsonify({'success': False, 'error': 'No elements with ID found in SVG'}), 400
                
            print(f"从SVG中提取的感知范围元素数量: {len(scope_elements)}")
            if debug_mode:
                debug_info['scope_elements_count'] = len(scope_elements)
                debug_info['scope_elements'] = scope_elements
            
            # 处理每个modify_element组
            # 创建一个包含所有需要修改的元素ID的列表
            all_modify_ids = []
            
            for element_group in modify_elements:
                # 获取IDs数组和属性
                element_ids = element_group.get('ids', [])
                attributes = element_group.get('attributes', {})
                
                if not element_ids or not attributes:
                    continue
                
                # 将当前组的IDs添加到所有修改ID的列表中
                all_modify_ids.extend(element_ids)
                
                # 处理特殊属性名转换
                processed_attributes = {}
                for attr_name, attr_value in attributes.items():
                    # 处理stroke相关属性
                    if attr_name in ['stroke lightness', 'stroke hue', 'stroke saturation']:
                        # 将stroke颜色属性转换为stroke属性
                        processed_attributes['stroke'] = attr_value
                    # 处理fill相关属性
                    elif attr_name in ['fill lightness', 'fill hue', 'fill saturation']:
                        # 将fill颜色属性转换为fill属性
                        processed_attributes['fill'] = attr_value
                    else:
                        # 其他属性保持不变
                        processed_attributes[attr_name] = attr_value
                
                # 对每个ID应用处理后的属性
                for element_id in element_ids:
                    # 查找元素
                    element = soup.find(id=element_id)
                    if not element:
                        print(f"未找到ID为{element_id}的元素")
                        continue
                    
                    # 修改属性
                    for attr_name, attr_value in processed_attributes.items():
                        # 特殊处理stroke-width属性
                        if attr_name == 'stroke-width':
                            # 获取原始stroke-width值，默认为1
                            original_width = element.get('stroke-width', '1')
                            
                            # 如果有单位（如px），去掉单位
                            if original_width.endswith('px'):
                                original_width = original_width[:-2]
                            
                            try:
                                # 将原始值转为数字并加上新的值
                                new_width = float(original_width) + float(attr_value)
                                element[attr_name] = f"{new_width}"
                            except ValueError:
                                print(f"无法处理stroke-width: {original_width} + {attr_value}")
                                element[attr_name] = f"{attr_value}"
                        
                        # 特殊处理area属性（调整元素尺寸）
                        elif attr_name == 'area':
                            # 根据元素类型不同，调整方式也不同
                            tag_name = element.name
                            
                            # 获取原始尺寸和位置
                            if tag_name == 'rect':
                                # 获取矩形的宽度和高度
                                try:
                                    width = float(element.get('width', '0'))
                                    height = float(element.get('height', '0'))
                                    x = float(element.get('x', '0'))
                                    y = float(element.get('y', '0'))
                                    
                                    # 计算缩放系数
                                    scale = float(attr_value)
                                    
                                    # 计算新的宽高
                                    new_width = width * scale
                                    new_height = height * scale
                                    
                                    # 计算新的位置，保持中心不变
                                    new_x = x - (new_width - width) / 2
                                    new_y = y - (new_height - height) / 2
                                    
                                    # 应用新的属性
                                    element['width'] = str(new_width)
                                    element['height'] = str(new_height)
                                    element['x'] = str(new_x)
                                    element['y'] = str(new_y)
                                except (ValueError, TypeError) as e:
                                    print(f"调整rect尺寸时出错: {str(e)}")
                                    
                            elif tag_name == 'circle':
                                # 获取圆的半径
                                try:
                                    r = float(element.get('r', '0'))
                                    cx = float(element.get('cx', '0'))
                                    cy = float(element.get('cy', '0'))
                                    
                                    # 计算缩放系数，圆形面积与半径平方成正比
                                    scale = float(attr_value)
                                    radius_scale = math.sqrt(scale)
                                    
                                    # 应用新的半径
                                    element['r'] = str(r * radius_scale)
                                except (ValueError, TypeError) as e:
                                    print(f"调整circle尺寸时出错: {str(e)}")
                                    
                            elif tag_name == 'ellipse':
                                # 获取椭圆的半径
                                try:
                                    rx = float(element.get('rx', '0'))
                                    ry = float(element.get('ry', '0'))
                                    cx = float(element.get('cx', '0'))
                                    cy = float(element.get('cy', '0'))
                                    
                                    # 计算缩放系数，椭圆面积与两个半径的乘积成正比
                                    scale = float(attr_value)
                                    radius_scale = math.sqrt(scale)
                                    
                                    # 应用新的半径
                                    element['rx'] = str(rx * radius_scale)
                                    element['ry'] = str(ry * radius_scale)
                                except (ValueError, TypeError) as e:
                                    print(f"调整ellipse尺寸时出错: {str(e)}")
                                    
                            # 其他类型元素暂不支持调整面积
                            else:
                                print(f"不支持调整{tag_name}元素的面积")
                        else:
                            # 其他属性直接设置
                            element[attr_name] = attr_value
            
            if debug_mode:
                debug_info['modify_ids'] = all_modify_ids
            
            # 将修改后的SVG转为字符串
            modified_svg_str = str(soup)
        except Exception as e:
            print(f"修改SVG元素属性时出错: {str(e)}")
            return jsonify({'success': False, 'error': f'Error modifying SVG: {str(e)}'}), 500
        
        # 创建临时文件
        temp_dir = tempfile.mkdtemp()
        temp_svg_path = os.path.join(temp_dir, 'modified.svg')
        
        # 保存修改后的SVG
        with open(temp_svg_path, 'w', encoding='utf-8') as f:
            f.write(modified_svg_str)
        
        # 使用featureCSV处理修改后的SVG
        temp_csv_path = os.path.join(temp_dir, 'features.csv')
        temp_svg_with_ids_path = os.path.join(temp_dir, 'svg_with_ids.svg')
        
        # 处理SVG提取特征，添加id和tag_name属性
        featureCSV.process_and_save_features(
            temp_svg_path, 
            temp_csv_path, 
            temp_svg_with_ids_path,
            add_ids=True, 
            add_tag_names=True
        )
        
        # 使用normalized_features处理特征
        temp_normalized_csv_path = os.path.join(temp_dir, 'normalized_features.csv')
        normalized_features.normalize_features(temp_csv_path, temp_normalized_csv_path)
        
        # 读取归一化的特征
        df = pd.read_csv(temp_normalized_csv_path)
        
        # 将数据转换为JSON格式，与SvgUploader.vue中的格式一致
        feature_data = []
        for _, row in df.iterrows():
            element_id = row['tag_name']
            features = row.iloc[1:].tolist()  # 跳过第一列tag_name
            feature_data.append({
                'id': element_id,
                'features': features
            })
            
        # 将特征数据添加到debug信息中
        if debug_mode:
            debug_info['normalized_features'] = feature_data
        
        # 计算显著性
        # 将所有节点分为高亮组和非高亮组（即modify_elements和其他）
        # 使用与SvgUploader.vue相同的处理逻辑
        
        # 将所有节点分为高亮组和非高亮组
        highlighted_features = []
        non_highlighted_features = []
        highlighted_ids = []
        non_highlighted_ids = []
        
        # 提取元素ID的最后部分并进行比较
        for item in feature_data:
            normalized_item_id = item['id']
            id_parts = normalized_item_id.split('/')
            item_id_last_part = id_parts[-1]
            
            # 判断是否是要修改的元素（在组内）
            if item_id_last_part in all_modify_ids:
                highlighted_features.append(item['features'])
                highlighted_ids.append(item_id_last_part)
            # 判断是否在感知范围内但不是要修改的元素（在组外）
            elif item_id_last_part in scope_elements:
                non_highlighted_features.append(item['features'])
                non_highlighted_ids.append(item_id_last_part)
        
        if debug_mode:
            debug_info['highlighted_count'] = len(highlighted_features)
            debug_info['non_highlighted_count'] = len(non_highlighted_features)
            debug_info['highlighted_ids'] = highlighted_ids
            debug_info['non_highlighted_ids'] = non_highlighted_ids
        
        # 如果没有足够的元素用于比较，返回默认值
        if len(highlighted_features) == 0 or len(non_highlighted_features) == 0:
            return jsonify({
                'success': True,
                'salience': 0.1,
                'debug_info': debug_info if debug_mode else None
            }), 200
        
        # 定义余弦相似度计算函数，与SvgUploader.vue一致
        def cosine_similarity_function(vec_a, vec_b):
            # 计算点积
            dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
            
            # 计算向量长度
            vec_a_magnitude = math.sqrt(sum(a * a for a in vec_a))
            vec_b_magnitude = math.sqrt(sum(b * b for b in vec_b))
            
            # 避免除以零
            if vec_a_magnitude == 0 or vec_b_magnitude == 0:
                return 0
            
            # 计算余弦相似度
            return dot_product / (vec_a_magnitude * vec_b_magnitude)
        
        # 计算组内元素的平均相似度
        intra_group_similarity = 1.0  # 默认设置为最大值
        
        # 如果组内有多个元素，计算它们之间的平均相似度
        if len(highlighted_features) > 1:
            similarity_sum = 0
            pair_count = 0
            
            # 计算组内所有元素对之间的相似度
            for i in range(len(highlighted_features)):
                for j in range(i + 1, len(highlighted_features)):
                    # 计算特征向量之间的余弦相似度
                    sim = cosine_similarity_function(highlighted_features[i], highlighted_features[j])
                    similarity_sum += sim
                    pair_count += 1
            
            # 计算平均相似度
            intra_group_similarity = similarity_sum / pair_count if pair_count > 0 else 1.0
        
        if debug_mode:
            debug_info['intra_group_similarity'] = intra_group_similarity
        
        # 计算组内与组外元素之间的平均相似度
        inter_group_similarity = 0
        inter_pair_count = 0
        
        # 计算每个组内元素与每个组外元素之间的相似度
        for i in range(len(highlighted_features)):
            for j in range(len(non_highlighted_features)):
                # 计算特征向量之间的余弦相似度
                sim = cosine_similarity_function(highlighted_features[i], non_highlighted_features[j])
                inter_group_similarity += sim
                inter_pair_count += 1
        
        # 计算平均相似度，避免除以零
        inter_group_similarity = inter_group_similarity / inter_pair_count if inter_pair_count > 0 else 0
        
        if debug_mode:
            debug_info['inter_group_similarity'] = inter_group_similarity
        
        # 避免除以零，如果组间相似度为0，设置显著性为最大值
        salience_score = intra_group_similarity / inter_group_similarity if inter_group_similarity > 0 else 1.0
        
        if debug_mode:
            debug_info['salience_score_base'] = salience_score
        
        # 考虑面积因素，参考SvgUploader.vue中的计算方法
        AREA_INDEX = 19  # bbox_fill_area 在特征向量中的索引是19
        
        # 计算所有元素的平均面积（包括高亮和非高亮元素）
        all_features = highlighted_features + non_highlighted_features
        all_elements_avg_area = sum(features[AREA_INDEX] for features in all_features) / len(all_features)
        
        # 计算高亮元素的平均面积
        highlighted_avg_area = sum(features[AREA_INDEX] for features in highlighted_features) / len(highlighted_features)
        
        # 使用所有元素平均面积的1.1倍作为阈值
        area_threshold = all_elements_avg_area * 1.1
        
        if debug_mode:
            debug_info['all_elements_avg_area'] = all_elements_avg_area
            debug_info['highlighted_avg_area'] = highlighted_avg_area
            debug_info['area_threshold'] = area_threshold
        
        # 如果高亮元素的平均面积小于阈值，显著降低显著性
        if highlighted_avg_area < area_threshold:
            salience_score = salience_score / 3
            if debug_mode:
                debug_info['applied_area_penalty'] = True
                debug_info['salience_score_after_area'] = salience_score
        else:
            if debug_mode:
                debug_info['applied_area_penalty'] = False
        
        # 检查是否与API聚类结果匹配
        add_bonus = True  # 默认添加额外分数
        
        # 对修改后的SVG重新运行聚类和子图检测，生成临时的API聚类结果
        try:
            # 创建临时目录用于存储聚类结果
            temp_clustering_dir = os.path.join(temp_dir, 'data')
            os.makedirs(temp_clustering_dir, exist_ok=True)
            
            # 准备数据目录
            temp_subgraphs_dir = os.path.join(temp_clustering_dir, 'subgraphs')
            os.makedirs(temp_subgraphs_dir, exist_ok=True)
            
            # 转为CSV为JSON
            temp_init_json = os.path.join(temp_clustering_dir, 'init_json.json')
            temp_normalized_init_json = os.path.join(temp_clustering_dir, 'normalized_init_json.json')
            
            featureCSV.process_csv_to_json(temp_csv_path, temp_init_json)
            featureCSV.process_csv_to_json(temp_normalized_csv_path, temp_normalized_init_json)
            
            # 运行聚类
            temp_community_data = os.path.join(temp_clustering_dir, 'community_data_mult.json')
            temp_features_data = os.path.join(temp_clustering_dir, 'cluster_features.json')
            
            run_clustering(
                temp_normalized_csv_path,
                temp_community_data,
                temp_features_data
            )
            
            # 运行子图检测
            run_subgraph_detection(
                features_json_path=temp_features_data,
                output_dir=temp_clustering_dir,
                clustering_method='gmm',
                subgraph_dimensions=[[0], [1], [2], [3], [0,1], [0,2], [0,3], [1,2], [1,3], [2,3],
                                   [0,1,2], [0,1,3], [0,2,3], [1,2,3], [0,1,2,3]]
            )
            
            # 读取子图维度信息
            temp_subgraph_all_path = os.path.join(temp_clustering_dir, 'subgraphs', 'subgraph_dimension_all.json')
            
            if os.path.exists(temp_subgraph_all_path):
                with open(temp_subgraph_all_path, 'r', encoding='utf-8') as f:
                    subgraph_data = json.load(f)
                
                # 获取所有核心聚类
                api_clusters = []
                if 'core_clusters' in subgraph_data:
                    for cluster in subgraph_data['core_clusters']:
                        nodes = []
                        for node_id in cluster['core_nodes']:
                            # 提取ID最后部分
                            node_id_last_part = node_id.split('/')[-1]
                            nodes.append(node_id_last_part)
                        
                        if nodes:
                            api_clusters.append(sorted(nodes))
                
                # 将高亮ID排序，以便进行比较
                sorted_highlighted_ids = sorted(highlighted_ids)
                
                # 检查高亮组是否与任何API聚类完全匹配
                for cluster in api_clusters:
                    if len(cluster) == len(sorted_highlighted_ids):
                        match = True
                        for i in range(len(cluster)):
                            if cluster[i] != sorted_highlighted_ids[i]:
                                match = False
                                break
                        
                        if match:
                            add_bonus = False
                            if debug_mode:
                                debug_info['cluster_match_found'] = True
                                debug_info['matched_cluster'] = cluster
                            break
                
                if debug_mode:
                    debug_info['api_clusters_count'] = len(api_clusters)
                    debug_info['highlighted_group'] = sorted_highlighted_ids
                    if not debug_info.get('cluster_match_found', False):
                        debug_info['all_api_clusters'] = api_clusters
            else:
                if debug_mode:
                    debug_info['subgraph_file_missing'] = True
                    debug_info['expected_path'] = temp_subgraph_all_path
            
        except Exception as e:
            print(f"运行API聚类和子图检测时出错: {str(e)}")
            print(f"错误堆栈: {traceback.format_exc()}")
            # 出错时仍然添加额外分数
            add_bonus = True
            if debug_mode:
                debug_info['clustering_error'] = str(e)
        
        # 只有在需要时添加额外显著性分数
        if add_bonus:
            salience_score += 0.4
            if debug_mode:
                debug_info['bonus_added'] = True
        else:
            if debug_mode:
                debug_info['bonus_added'] = False
        
        if debug_mode:
            debug_info['salience_score_after_bonus'] = salience_score
        
        # 使用与SvgUploader.vue完全相同的sigmoid函数进行平滑映射
        normalized_score = min(max(1 / (0.8 + math.exp(-salience_score)), 0), 1)
        
        if debug_mode:
            debug_info['final_normalized_score'] = normalized_score
            # 计算与目标值的差异
            debug_info['target_score'] = 0.67177
            debug_info['difference'] = abs(normalized_score - 0.67177)
            # 获取与目标显著性相对应的原始显著性分值
            target_salience_raw = -math.log(1/0.67177 - 0.8)
            debug_info['target_salience_raw'] = target_salience_raw
            
        # 清理临时文件
        shutil.rmtree(temp_dir)
        
        response_data = {
            'success': True,
            'salience': normalized_score
        }
        
        if debug_mode:
            response_data['debug_info'] = debug_info
            
        return jsonify(response_data), 200
        
    except Exception as e:
        print(f"计算显著性时出错: {str(e)}")
        print(f"错误堆栈: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
