import os
import json

def analyze_questionnaire_data():
    # 文件夹路径
    folder_path = 'questionnaire/public/Test_QuestionnaireData'
    
    # 初始化统计变量
    total_groups = 0  # 总标记组数
    person_count = 0  # 人数
    images_per_person = {}  # 每个人标注的图数
    
    # 遍历文件夹中的所有JSON文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            person_count += 1
            file_path = os.path.join(folder_path, filename)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    
                    # 获取学生ID
                    student_id = data.get('formData', {}).get('studentid', 'unknown')
                    
                    # 初始化该学生的图片数
                    images_per_person[student_id] = 0
                    
                    # 分析steps数据
                    steps = data.get('steps', [])
                    for step in steps:
                        if 'groups' in step:
                            # 该学生标注的图片数+1
                            images_per_person[student_id] += 1
                            # 累加该图片的标记组数
                            total_groups += len(step['groups'])
            
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {e}")
    
    # 计算总图片数
    total_images = sum(images_per_person.values())
    
    # 打印结果
    print(f"标注的总标记组数: {total_groups}")
    print(f"参与标注的人数: {person_count}")
    print(f"标注的总图片数: {total_images}")
    print(f"平均每人标注图片数: {total_images / person_count if person_count > 0 else 0:.2f}")
    print(f"平均每张图的标记组数: {total_groups / total_images if total_images > 0 else 0:.2f}")
    
    # 打印每个人标注的图片数
    print("\n每个人标注的图片数:")
    for student_id, image_count in images_per_person.items():
        print(f"{student_id}: {image_count}张图")

if __name__ == "__main__":
    analyze_questionnaire_data()
