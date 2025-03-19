import os
import cairosvg
from PIL import Image
import io

# 定义输入和输出文件夹
input_folders = [
    'Test_svg',
    'Train_svg'
]
output_folder = 'png_output'

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 设置所有输出PNG的统一尺寸
output_width = 80
output_height = 60

# 用于跟踪已处理的文件名，避免重名
processed_files = {}

def convert_svg_to_png(svg_path, png_path, width=output_width, height=output_height):
    """将SVG文件转换为指定尺寸的PNG文件"""
    # 使用cairosvg将SVG转换为PNG字节流
    png_data = cairosvg.svg2png(url=svg_path)
    
    # 使用PIL打开PNG字节流
    image = Image.open(io.BytesIO(png_data))
    
    # 调整图像大小
    image = image.resize((width, height), Image.LANCZOS)
    
    # 保存调整大小后的图像
    image.save(png_path)
    
    print(f"已转换: {svg_path} -> {png_path}")

# 处理所有文件夹中的SVG文件
total_converted = 0
for folder_index, folder_path in enumerate(input_folders):
    folder_name = os.path.basename(folder_path)
    
    # 遍历当前文件夹中的所有SVG文件
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith('.svg'):
            svg_path = os.path.join(folder_path, file_name)
            
            # 创建唯一的输出文件名（添加文件夹前缀）
            base_name = os.path.splitext(file_name)[0]
            
            # 如果文件名已存在，则添加文件夹名作为前缀
            if base_name in processed_files:
                new_base_name = f"{folder_name}_{base_name}"
            else:
                new_base_name = base_name
                processed_files[base_name] = True
            
            png_path = os.path.join(output_folder, f"{new_base_name}.png")
            
            # 转换SVG为PNG
            try:
                convert_svg_to_png(svg_path, png_path)
                total_converted += 1
            except Exception as e:
                print(f"转换失败 {svg_path}: {str(e)}")

print(f"转换完成! 共转换 {total_converted} 个文件到 {output_folder}")
