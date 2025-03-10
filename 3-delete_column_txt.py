import numpy as np

def remove_columns_from_txt(input_file, output_file):
    # 读取文件数据
    data = np.loadtxt(input_file)
    
    # 删除指定的列（索引从0开始）
    # 这里删除第1列（索引0）、第8列（索引7）、第15列（索引14）
    data_removed = np.delete(data, [0, 7, 14], axis=1)
    # data_removed = np.delete(data, [18, 19], axis=1)
    
    # 将删除后的数据保存到输出文件
    np.savetxt(output_file, data_removed, fmt='%.6f')

    print(f"数据已处理并保存到: {output_file}")

# 调用函数，文件路径按需修改
remove_columns_from_txt('joint_data_abs_0102.txt', 'joint_data_abs_0104.txt')
