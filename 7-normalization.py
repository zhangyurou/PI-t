import numpy as np

# 假设你的原始数据文件名为 'data.txt'
# 如果数据之间是空格/Tab 分隔，np.loadtxt 默认即可处理；
# 如果是其他分隔符，可通过 delimiter 参数指定，例如 delimiter=',' 表示逗号分隔
data = np.loadtxt('pusht_PINN_LSTM_0220.txt')

# data 的形状应为 (114962, 6)
print('数据形状:', data.shape)

# 分别求每列的最小值和最大值
min_vals = np.min(data, axis=0)
max_vals = np.max(data, axis=0)
print(min_vals, max_vals)

# Min-Max 归一化： (x - min) / (max - min)
# 注意要确保 max_vals != min_vals，否则会产生除 0 错误
normalized_data = (data - min_vals) / (max_vals - min_vals)

# 保存归一化后的结果到新文件，如 'normalized_data.txt'
np.savetxt('norm_pusht_PINN_LSTM_0220.txt', normalized_data, fmt='%.6f')


# import numpy as np

# min_vals, max_vals 是在做归一化时保存下来的原始列最小值和最大值
# 例如:
# min_vals = np.min(original_data, axis=0)
# max_vals = np.max(original_data, axis=0)

'''
# 加载或准备好已经归一化的数据
normalized_data = np.loadtxt('normalized_data.txt')  # shape (N, D)，例如 (114962, 6)

# 执行反归一化
denormalized_data = normalized_data * (max_vals - min_vals) + min_vals

# 保存结果
np.savetxt('denormalized_data.txt', denormalized_data, fmt='%.6f')
'''