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
        init_json = os.path.join(app.config['DATA_FOLDER'], 'init_json.json')
        normalized_init_json = os.path.join(app.config['DATA_FOLDER'], 'normalized_init_json.json')
        # 调用 cluster.py 进行聚类
        community_data_path = os.path.join(app.config['DATA_FOLDER'], 'community_data_mult.json')
        features_data_path = os.path.join(app.config['DATA_FOLDER'], 'cluster_features.json')
        # 调用 normalized_features.py 进行归一化处理
        normalized_csv_path = os.path.join(app.config['DATA_FOLDER'], 'normalized_features.csv')


        global max_eps, epss  # 声明使用全局变量

        featureCSV.process_and_save_features(file_path, output_csv_path, output_svg_with_ids_path)
        normalized_features.normalize_features(output_csv_path, normalized_csv_path)

        featureCSV.process_csv_to_json(output_csv_path, init_json)
        featureCSV.process_csv_to_json(normalized_csv_path, normalized_init_json)

        calculator = EquivalentWeightsCalculator(model_path="static/modules/checkpoint_200.tar")
        calculator.compute_and_save_equivalent_weights(normalized_csv_path, output_file_avg='static/data/average_equivalent_mapping.json', output_file_all='static/data/equivalent_weights_by_tag.json')

        epss, max_eps= get_eps(normalized_csv_path, community_data_path, features_data_path)
        run_clustering(normalized_csv_path, community_data_path, features_data_path, max_eps)

        # draw_element_nodes_with_lines(init_json, features_data_path)

        return jsonify({
            'success': 'File uploaded and processed successfully',
            'svg_file': file_path,
            'csv_file': output_csv_path,
            'normalized_csv_file': normalized_csv_path,
            'community_data_mult': community_data_path,
            'cluster_features': features_data_path,
            'svg_with_ids_file': f'/{output_svg_with_ids_path}'
        }), 200
    else:
        return jsonify({'error': 'Invalid file type'}), 400

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
    # 确保 dimension 参数在有效范围内（假设有4个特征维度，即0到3）
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


if __name__ == '__main__':
    app.run(debug=True, port=5000)
