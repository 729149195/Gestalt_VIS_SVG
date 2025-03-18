import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np

def create_visualizations(df):
    # 设置中文字体（如果你的系统支持）
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 1. 年龄分布 - 直方图
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='age', bins=10)
    plt.title('年龄分布')
    plt.xlabel('年龄')
    plt.ylabel('人数')
    plt.savefig('age_distribution.png')
    plt.close()

    # 2. 性别分布 - 饼图
    plt.figure(figsize=(8, 8))
    gender_counts = df['gender'].value_counts()
    plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%')
    plt.title('性别分布')
    plt.savefig('gender_distribution.png')
    plt.close()

    # 3. 可视化经验分布 - 饼图
    plt.figure(figsize=(8, 8))
    exp_counts = df['visualizationExperience'].value_counts()
    plt.pie(exp_counts, labels=exp_counts.index, autopct='%1.1f%%')
    plt.title('可视化经验分布')
    plt.savefig('visualization_experience_distribution.png')
    plt.close()

    # 4. 完成时间分布 - 箱型图
    # 首先将duration转换为分钟数
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
    plt.savefig('duration_distribution.png')
    plt.close()

    # 5. 年龄与完成时间的关系 - 散点图
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='age', y='duration_minutes')
    plt.title('年龄与完成时间的关系')
    plt.xlabel('年龄')
    plt.ylabel('完成时间（分钟）')
    plt.savefig('age_duration_relationship.png')
    plt.close()

    # 6. 性别与完成时间的关系 - 箱型图
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='gender', y='duration_minutes')
    plt.title('性别与完成时间的关系')
    plt.xlabel('性别')
    plt.ylabel('完成时间（分钟）')
    plt.savefig('gender_duration_relationship.png')
    plt.close()

if __name__ == "__main__":
    # 读取Excel文件
    df = pd.read_excel('student_statistics.xlsx')
    create_visualizations(df)