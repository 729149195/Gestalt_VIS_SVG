from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO
import os
from static.modules import featureCSV
from static.modules import normalized_features
from static.modules.cluster import main as run_clustering

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


@app.route('/')
def hello_world():
    return 'Hello World!'


# 文件上传的路由
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and file.filename.endswith('.svg'):
        # 保存上传的文件
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # 处理 SVG 文件
        output_csv_path = os.path.join(app.config['DATA_FOLDER'], 'features.csv')
        output_svg_with_ids_path = os.path.join(app.config['DATA_FOLDER'], 'svg_with_ids.svg')

        # 调用 featureCSV.py 中的函数
        featureCSV.process_and_save_features(file_path, output_csv_path, output_svg_with_ids_path)

        # 调用 normalized_features.py 进行归一化处理
        normalized_csv_path = os.path.join(app.config['DATA_FOLDER'], 'normalized_features.csv')
        normalized_features.normalize_features(output_csv_path, normalized_csv_path)

        # 调用 cluster.py 进行聚类
        community_data_path = os.path.join(app.config['DATA_FOLDER'], 'community_data_mult.json')
        probabilities_data_path = os.path.join(app.config['DATA_FOLDER'], 'cluster_probabilities.json')
        run_clustering(normalized_csv_path, community_data_path, probabilities_data_path)

        return jsonify({
            'success': 'File uploaded and processed successfully',
            'svg_file': file_path,
            'csv_file': output_csv_path,
            'normalized_csv_file': normalized_csv_path,
            'community_data_mult': community_data_path,
            'cluster_probabilities': probabilities_data_path,
            'svg_with_ids_file': f'/{output_svg_with_ids_path}'
        }), 200
    else:
        return jsonify({'error': 'Invalid file type'}), 400


# 获取生成的 SVG 文件内容
@app.route('/get_svg', methods=['GET'])
def get_svg():
    svg_file_path = os.path.join(app.config['DATA_FOLDER'], 'svg_with_ids.svg')

    if os.path.exists(svg_file_path):
        with open(svg_file_path, 'r', encoding='utf-8') as svg_file:
            svg_content = svg_file.read()
        return svg_content, 200, {'Content-Type': 'image/svg+xml'}
    else:
        return jsonify({'error': 'SVG file not found'}), 404


@app.route('/community_data_mult', methods=['GET'])
def get_community_data_mult():
    community_data_path = os.path.join(app.config['DATA_FOLDER'], 'community_data_mult.json')

    if os.path.exists(community_data_path):
        with open(community_data_path, 'r', encoding='utf-8') as json_file:
            return json_file.read(), 200, {'Content-Type': 'application/json'}
    else:
        return jsonify({'error': 'community_data_mult.json file not found'}), 404


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


@app.route('/cluster_probabilities', methods=['GET'])
def cluster_probabilities():
    community_data_path = os.path.join(app.config['DATA_FOLDER'], 'cluster_probabilities.json')

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


@app.route('/top_position', methods=['GET'])
def top_position():
    community_data_path = os.path.join(app.config['DATA_FOLDER'], 'Top_data.json')

    if os.path.exists(community_data_path):
        with open(community_data_path, 'r', encoding='utf-8') as json_file:
            return json_file.read(), 200, {'Content-Type': 'application/json'}
    else:
        return jsonify({'error': 'community_data_mult.json file not found'}), 404


if __name__ == '__main__':
    app.run(debug=True, port=5000)
