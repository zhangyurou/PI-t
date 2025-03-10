# 定义文件路径
input_file = 'norm_pusht_PINN_LSTM_0220.txt'  # 替换为你的源文件名或路径
train_file = 'training_pusht_set26.txt'     # 训练集文件名
test_file = 'testing_pusht_set26.txt'       # 测试集文件名

# 定义切分点
train_size = 20500

try:
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(train_file, 'w', encoding='utf-8') as fout_train, \
         open(test_file, 'w', encoding='utf-8') as fout_test:
        
        for line_num, line in enumerate(fin, start=1):
            if line_num <= train_size:
                fout_train.write(line)
            else:
                fout_test.write(line)
        
        print(f"成功将前 {train_size} 行写入 {train_file}，剩余 {line_num - train_size} 行写入 {test_file}。")
                
except FileNotFoundError:
    print(f"未找到输入文件：{input_file}")
except Exception as e:
    print(f"发生错误：{e}")
