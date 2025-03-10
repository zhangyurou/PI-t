# import zarr

# path = '.'
# zarr_arr = zarr.open(f'{path}/zarr_data.zarr', mode='r')

# print(zarr_arr)


import zarr
import os
import numpy as np

path = '.'
# zarr_arr = zarr.open(f'{path}/data.zarr', mode='r')
zarr_arr = zarr.open(f'{path}/multimodal_push_seed_abs.zarr', mode='r')

# print(zarr_arr)
# 列出根组下的所有子组和数组
print(zarr_arr.tree())
# print(zarr.load('{path}/data.zarr'))  # load只能加载数组,不能加载group
# print(os.path.exists(f'{path}/data.zarr'))
print(os.path.exists(f'{path}/multimodal_push_seed_abs.zarr'))


b_data = zarr_arr['data/action'][:]
print(b_data)

'''
action_data = zarr_arr['data/action'][:]
print(action_data)

obs_data = zarr_arr['data/obs'][:]
print(obs_data)

meta_data = zarr_arr['meta/episode_ends'][:]
print(meta_data)

# 打开文件
with open('action_data.txt', 'w') as f:
    for row in action_data:
        # 将数组的每一行转换为字符串并写入文件
        f.write(' '.join(map(str, row)) + '\n')


# 打开文件
with open('obs_data.txt', 'w') as f:
    for row in obs_data:
        # 将数组的每一行转换为字符串并写入文件
        f.write(' '.join(map(str, row)) + '\n')


# 打开文件
with open('meta_data.txt', 'w') as f:
    for i in range(0, len(meta_data), 10):
        # 每10个元素作为一行写入
        f.write(' '.join(map(str, meta_data[i:i+10])) + '\n')
'''
        

# 打开文件
with open('multimodal_push_seed_abs.txt', 'w') as f:
    for row in b_data:
        # 将数组的每一行转换为字符串并写入文件
        f.write(' '.join(map(str, row)) + '\n')