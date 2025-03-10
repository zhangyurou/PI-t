import numpy as np

def merge_txt_files(file1, file2, output_file):
    # 读取第一个文件
    data1 = np.loadtxt(file1)
    
    # 读取第二个文件
    data2 = np.loadtxt(file2)
    
    # 检查两个文件的行数是否一致
    if data1.shape[0] != data2.shape[0]:
        raise ValueError(f"文件行数不一致: {data1.shape[0]} != {data2.shape[0]}")
    
    # 对两个文件的数据按列拼接
    merged_data = np.hstack((data1, data2))
    
    # 将拼接后的数据保存到输出文件
    np.savetxt(output_file, merged_data, fmt='%.6f')

    print(f"文件成功拼接并保存到: {output_file}")

# 调用函数，文件路径按需修改
merge_txt_files('pusht_xarm.txt', 'output_torque_pusht_action_delta.txt', 'pusht_PINN_LSTM_0220.txt')
