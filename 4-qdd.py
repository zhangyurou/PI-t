import pandas as pd
import numpy as np

# 设置文件路径
input_file_path = 'pusht_action_delta_joint_0220.txt'    # 替换为您的原始 TXT 文件路径
output_file_path = 'pusht_qdd'  # 替换为您希望保存的新 TXT 文件路径

# 时间间隔（秒）
delta_t = 0.39

# 读取数据
try:
    # 假设数据是以空格或制表符分隔的。如果是其他分隔符（如逗号），请调整 'sep' 参数
    data = pd.read_csv(input_file_path, sep=r'\s+', header=None)
except Exception as e:
    print(f"读取文件时出错: {e}")
    exit(1)

# 检查数据是否至少有12列
if data.shape[1] < 12:
    print("数据列不足12列，请检查文件内容。")
    exit(1)

# 提取关节位置（第1-6列）和速度（第7-12列）
positions = data.iloc[:, 0:6]  # 列0到列5
velocities = data.iloc[:, 6:12]  # 列6到列11

# 计算加速度
# 使用差分方法计算加速度：a(t) = (v(t) - v(t-1)) / delta_t
# 由于差分导致数据行数减少1，因此在最前面插入NaN或0以保持数据对齐
accelerations = velocities.diff().divide(delta_t)
accelerations.iloc[0] = 0  # 将第一行加速度设置为0，或者您可以选择设置为NaN

# 确保加速度的数据类型为数值型
accelerations = accelerations.apply(pd.to_numeric, errors='coerce')

# 合并位置、速度和加速度数据
# 重新排列列顺序为：位置1-6，速度7-12，加速度13-18
combined_data = pd.concat([positions, velocities, accelerations], axis=1)

# 检查合并后的数据是否有NaN（可选）
if combined_data.isnull().values.any():
    print("警告：合并后的数据包含缺失值（NaN）。请检查原始数据和计算过程。")

# 将合并后的数据写入新的 TXT 文件
try:
    # 使用空格作为分隔符，并且不写入行索引和列标题
    combined_data.to_csv(output_file_path, sep=' ', header=False, index=False, float_format='%.6f')
    print(f"数据已成功写入 {output_file_path}")
except Exception as e:
    print(f"写入文件时出错: {e}")
    exit(1)
