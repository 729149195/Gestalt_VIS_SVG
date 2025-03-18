import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

def create_visualizations(df, writer):
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建新的工作表用于存放图表
    worksheet = writer.book.add_worksheet('可视化')
    current_row = 0
    
    # 1. 年龄分布
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='age', bins=10)
    plt.title('年龄分布')
    plt.xlabel('年龄')
    plt.ylabel('人数')
    
    # 将图表保存到内存中
    img_buf = BytesIO()
    plt.savefig(img_buf, format='png', bbox_inches='tight', dpi=300)
    img_buf.seek(0)
    worksheet.insert_image(current_row, 0, '', {'image_data': img_buf})
    current_row += 30
    plt.close()

    # 2. 性别分布
    plt.figure(figsize=(8, 8))
    gender_counts = df['gender'].value_counts()
    plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%')
    plt.title('性别分布')
    
    img_buf = BytesIO()
    plt.savefig(img_buf, format='png', bbox_inches='tight', dpi=300)
    img_buf.seek(0)
    worksheet.insert_image(current_row, 0, '', {'image_data': img_buf})
    current_row += 30
    plt.close()

    # 3. 可视化经验分布
    plt.figure(figsize=(8, 8))
    exp_counts = df['visualizationExperience'].value_counts()
    plt.pie(exp_counts, labels=exp_counts.index, autopct='%1.1f%%')
    plt.title('可视化经验分布')
    
    img_buf = BytesIO()
    plt.savefig(img_buf, format='png', bbox_inches='tight', dpi=300)
    img_buf.seek(0)
    worksheet.insert_image(current_row, 0, '', {'image_data': img_buf})
    current_row += 30
    plt.close()

    # 4. 完成时间分布
    def convert_duration_to_minutes(duration_str):
        parts = duration_str.split()
        minutes = float(parts[0])
        if len(parts) > 2:
            seconds = float(parts[2])
            minutes += seconds / 60
        return minutes

    df['duration_minutes'] = df['duration'].apply(convert_duration_to_minutes)

    plt.figure(figsize=(10, 6))
    sns.boxplot(y=df['duration_minutes'])
    plt.title('完成时间分布')
    plt.ylabel('时间（分钟）')
    
    img_buf = BytesIO()
    plt.savefig(img_buf, format='png', bbox_inches='tight', dpi=300)
    img_buf.seek(0)
    worksheet.insert_image(current_row, 0, '', {'image_data': img_buf})
    plt.close()

def collect_student_ids():
    results = []
    folder_path = "public/QuestionnaireData2"
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                student_info = {
                    'studentid': data['formData']['studentid'],
                    'age': data['formData']['age'],
                    'gender': data['formData']['gender'],
                    'visualimpairment': data['formData']['visualimpairment'],
                    'visualizationExperience': data['formData']['visualizationExperience'],
                    'duration': data['duration']
                }
                results.append(student_info)
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {str(e)}")
    
    df = pd.DataFrame(results)
    df = df.sort_values('studentid')
    
    # 使用ExcelWriter来创建多个工作表
    output_file = 'student_statistics.xlsx'
    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
        # 将数据保存到第一个工作表
        df.to_excel(writer, sheet_name='数据', index=False)
        
        # 创建可视化并保存到第二个工作表
        create_visualizations(df, writer)
    
    print(f"统计结果和可视化已保存到 {output_file}")
    
    # 打印基本统计信息
    print("\n基本统计信息：")
    print(f"总人数: {len(df)}")
    print("\n性别分布:")
    print(df['gender'].value_counts())
    print("\n可视化经验分布:")
    print(df['visualizationExperience'].value_counts())

if __name__ == "__main__":
    collect_student_ids()