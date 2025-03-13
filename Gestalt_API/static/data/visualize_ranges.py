import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import matplotlib as mpl
from matplotlib.font_manager import FontProperties

def visualize_feature_ranges():
    """
    可视化feature_ranges.csv中的Q1和Q3范围，生成水平条形图和热图，保持原始顺序
    """
    # 设置中文字体支持
    # 尝试设置中文字体，如果失败则使用默认字体
    try:
        # 尝试使用系统中的中文字体
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        
        # 检查是否有可用的中文字体
        font_names = [f.name for f in mpl.font_manager.fontManager.ttflist]
        chinese_fonts = [f for f in font_names if 'microsoft' in f.lower() or 'simsun' in f.lower() or 'simhei' in f.lower()]
        
        if chinese_fonts:
            print(f"找到可用的中文字体: {chinese_fonts[:3]}...")
        else:
            print("警告: 未找到中文字体，图表中的中文可能无法正确显示")
    except Exception as e:
        print(f"设置中文字体时出错: {e}")
        print("将使用默认字体，中文可能无法正确显示")
    
    # 读取feature_ranges.csv文件
    csv_path = 'feature_ranges.csv'
    if not os.path.exists(csv_path):
        print(f"错误：找不到文件 {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    print(f"读取了 {len(df)} 个特征的范围数据")
    
    # 创建输出目录
    output_dir = 'range_plots'
    os.makedirs(output_dir, exist_ok=True)
    
    # 计算IQR (Interquartile Range)，但不用于排序
    df['iqr'] = df['q3_value'] - df['q1_value']
    
    # 使用原始顺序，不进行排序
    df_original = df.copy()
    
    # 1. 创建水平条形图，显示每个特征的Q1-Q3范围
    plt.figure(figsize=(14, 10))  # 增加宽度以容纳左侧的特征名称
    
    # 创建水平条形图
    y_pos = np.arange(len(df_original))
    
    # 绘制从Q1到Q3的条形
    plt.barh(y_pos, df_original['q3_value'] - df_original['q1_value'], 
             left=df_original['q1_value'], height=0.7, 
             color='skyblue', alpha=0.8)
    
    # 在左侧添加特征名称，而不是在条形内部
    plt.yticks(y_pos, df_original['column'], fontsize=9)
    
    # 添加Q1和Q3的点
    plt.scatter(df_original['q1_value'], y_pos, color='blue', s=50, label='Q1')
    plt.scatter(df_original['q3_value'], y_pos, color='red', s=50, label='Q3')
    
    # 添加图表标题和标签
    plt.title('Features Q1-Q3 Range (Original Order)', fontsize=14)
    plt.xlabel('Value', fontsize=12)
    plt.xlim(0, 1.05)  # 设置x轴范围，假设所有值都在0-1之间
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.legend()
    
    # 保存图表
    range_plot_path = os.path.join(output_dir, 'feature_ranges_plot.png')
    plt.tight_layout()
    plt.savefig(range_plot_path, dpi=300)
    plt.close()
    print(f"保存范围图表到: {range_plot_path}")
    
    # 2. 创建热图，显示所有特征的Q1和Q3值
    # 准备热图数据
    heatmap_data = df_original.set_index('column')[['q1_value', 'q3_value']]
    
    plt.figure(figsize=(10, 12))
    sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu', fmt='.2f', 
                linewidths=.5, cbar_kws={'label': 'Value'})
    plt.title('Feature Q1 and Q3 Values Heatmap (Original Order)', fontsize=14)
    plt.tight_layout()
    
    # 保存热图
    heatmap_path = os.path.join(output_dir, 'feature_ranges_heatmap.png')
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    print(f"保存热图到: {heatmap_path}")
    
    # 3. 创建特征分布图，显示每个特征的Q1、Q3和IQR
    plt.figure(figsize=(14, 8))  # 增加宽度以容纳标签
    
    # 绘制IQR条形图
    bars = plt.bar(df_original['column'], df_original['iqr'], color='lightblue', alpha=0.7)
    
    # 添加Q1和Q3的点
    plt.scatter(range(len(df_original)), df_original['q1_value'], color='blue', s=50, label='Q1')
    plt.scatter(range(len(df_original)), df_original['q3_value'], color='red', s=50, label='Q3')
    
    # 连接Q1和Q3的线
    for i in range(len(df_original)):
        plt.plot([i, i], [df_original.iloc[i]['q1_value'], df_original.iloc[i]['q3_value']], 
                 color='black', linestyle='-', linewidth=1.5)
    
    # 添加图表标题和标签
    plt.title('Feature Q1, Q3 and IQR Distribution (Original Order)', fontsize=14)
    plt.xlabel('Feature', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.xticks(rotation=90)  # 旋转x轴标签以避免重叠
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    
    # 保存图表
    distribution_path = os.path.join(output_dir, 'feature_distribution_plot.png')
    plt.tight_layout()
    plt.savefig(distribution_path, dpi=300)
    plt.close()
    print(f"保存分布图到: {distribution_path}")
    
    # 4. 创建一个HTML文件来展示所有图表
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Feature Range Visualization</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            h1, h2 {
                color: #333;
                text-align: center;
            }
            .image-container {
                margin: 30px 0;
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            img {
                max-width: 100%;
                height: auto;
                display: block;
                margin: 0 auto;
            }
            .description {
                margin: 15px 0;
                line-height: 1.5;
                color: #555;
            }
        </style>
    </head>
    <body>
        <h1>Feature Range Visualization</h1>
        
        <div class="image-container">
            <h2>Feature Q1-Q3 Range (Original Order)</h2>
            <img src="range_plots/feature_ranges_plot.png" alt="Feature Ranges Plot">
            <div class="description">
                This chart shows the range between Q1 (first quartile) and Q3 (third quartile) for each feature, 
                in their original order from the CSV file. Blue dots represent Q1 values and red dots represent Q3 values.
                Feature names are displayed on the left side of the chart.
            </div>
        </div>
        
        <div class="image-container">
            <h2>Feature Q1 and Q3 Values Heatmap</h2>
            <img src="range_plots/feature_ranges_heatmap.png" alt="Feature Ranges Heatmap">
            <div class="description">
                This heatmap visualizes the Q1 and Q3 values for each feature. 
                Darker colors indicate higher values. The features are shown in their original order.
            </div>
        </div>
        
        <div class="image-container">
            <h2>Feature Q1, Q3 and IQR Distribution</h2>
            <img src="range_plots/feature_distribution_plot.png" alt="Feature Distribution Plot">
            <div class="description">
                This chart shows the distribution of Q1, Q3 values and the IQR for each feature. 
                The blue bars represent the IQR, while the blue and red dots represent Q1 and Q3 values respectively.
                The vertical lines connect the Q1 and Q3 values for each feature. Features are shown in their original order.
            </div>
        </div>
    </body>
    </html>
    """
    
    # 保存HTML文件
    html_path = 'feature_ranges_visualization.html'  # 直接保存在当前目录，而不是range_plots子目录
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"保存HTML可视化页面到: {html_path}")
    
    print("所有可视化已完成！")

if __name__ == "__main__":
    visualize_feature_ranges() 