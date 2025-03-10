import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import sympy as sp
import time
import logging
import datetime

# 获取当前日期并格式化
current_date = datetime.datetime.now().strftime("%Y-%m-%d")
log_filename = f"PINN-{current_date}-train-pusht-LSTM-mask-20.log"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用的设备: {device}")

# 配置基本日志设置
logging.basicConfig(
    level=logging.DEBUG,  # 设置最低日志级别
    format='%(asctime)s - %(levelname)s - %(message)s',  # 日志格式
    datefmt='%Y-%m-%d %H:%M:%S',  # 时间格式
    filename = log_filename,  # 日志文件名（如果需要输出到文件）
    filemode='w'  # 文件写入模式，'w' 表示覆盖，'a' 表示追加
)

# 如果需要同时输出到控制台，可以添加一个 StreamHandler
console = logging.StreamHandler()
console.setLevel(logging.INFO)  # 设置控制台日志级别
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

def apply_random_mask(data, mask_prob=0.3):
    """
    对输入数据中的关节角度和速度随机应用掩码，mask_prob 控制掩码概率。
    :param data: (batch_size, seq_len, feature_dim)
    :param mask_prob: 掩码概率 (0~1)
    :return: 掩码后的数据
    """
    mask = np.random.rand(*data.shape) > mask_prob  # 生成掩码矩阵
    return data * mask  # 仅部分元素被置零

# 数据读取和预处理
'''
def load_data_train(filename):
    # 读取数据
    data = np.loadtxt(filename)
    
    # 数据拆分
    joint_angles = data[:, 0:6]      # q1 - q6
    joint_velocities = data[:, 6:12] # dq1 - dq6
    # joint_torques = data[:, 12:18]   # tau1 - tau6
    end_effector_positions = data[:, 18:20] # x_end, y_end
    dynamics_tau = data[:, 20:26] # tau1 - tau6
    
    # 构建序列数据
    sequence_length = 12
    num_sequences = int((joint_angles.shape[0] - sequence_length) / 1) + 1
    print("num_sequences", num_sequences)
    
    # 初始化数据列表
    input_data = []
    output_data = []
    
    for i in range(num_sequences):  # 114951组  9581组
        # 输入特征
        s = i
        input_seq = np.hstack([
            joint_angles[s:s+sequence_length],
            joint_velocities[s:s+sequence_length],
            end_effector_positions[s:s+sequence_length]
        ])
        # 输出特征
        output_seq = np.hstack([
            end_effector_positions[s:s+sequence_length],
            # joint_torques[s:s+sequence_length],
            dynamics_tau[s:s+sequence_length]
        ])
    
        input_data.append(input_seq)
        output_data.append(output_seq)
        
    # 转换为 NumPy 数组
    input_data = np.array(input_data)
    # print("input_data", input_data.shape)
    output_data = np.array(output_data)
    # print("output_data", output_data.shape)

    return input_data, output_data
'''

# mask
def load_data_train(filename, apply_mask=True, mask_prob=0.3):
    """
    加载训练数据，并在训练时对关节角度和速度随机掩码
    """
    data = np.loadtxt(filename)
    joint_angles = data[:, 0:6]      
    joint_velocities = data[:, 6:12]  
    end_effector_positions = data[:, 18:20]
    dynamics_tau = data[:, 20:26]

    # 构造时序数据
    sequence_length = 12
    num_sequences = int((joint_angles.shape[0] - sequence_length) / 1) + 1
    print("num_sequences", num_sequences)

    input_data, output_data = [], []

    for i in range(num_sequences):
        s = i
        input_seq = np.hstack([
            joint_angles[s:s+sequence_length],
            joint_velocities[s:s+sequence_length],
            end_effector_positions[s:s+sequence_length]
        ])
        output_seq = np.hstack([
            end_effector_positions[s:s+sequence_length],
            dynamics_tau[s:s+sequence_length]
        ])

        # 应用随机掩码（仅对关节角度和速度）
        if apply_mask:
            input_seq[:, :12] = apply_random_mask(input_seq[:, :12], mask_prob=mask_prob)

        input_data.append(input_seq)
        output_data.append(output_seq)

    return np.array(input_data), np.array(output_data)


class PINN_LSTM(nn.Module):
    def __init__(self):
        super(PINN_LSTM, self).__init__()
        
        # LSTM层：输入维度14，隐藏状态维度128，序列长度为12（序列长度由输入数据决定）
        self.lstm = nn.LSTM(input_size=14, hidden_size=128, num_layers=2, batch_first=True)
        
        # 线性层，将LSTM的输出映射到目标输出维度 8
        self.fc = nn.Linear(128, 8)  # 输出每个时间步8个特征

    def forward(self, x):
        # x的形状是 (batch_size, seq_len, input_size)，例如 (256, 12, 14)
        
        # LSTM层的输出：lstm_out形状 (batch_size, seq_len, hidden_size)
        lstm_out, (hn, cn) = self.lstm(x)
        
        # 对LSTM输出的每个时间步都进行全连接层变换，输出形状为 (batch_size, seq_len, 8)
        output = self.fc(lstm_out)  # 经过线性层后，每个时间步的输出是 8 维
        print(output.shape)
        return output



'''
class PINN_LSTM(nn.Module):
    def __init__(self):
        super(PINN_LSTM, self).__init__()
        
        # LSTM层：输入维度14，隐藏状态维度128，LSTM层数为3
        self.lstm = nn.LSTM(input_size=14, hidden_size=64, num_layers=3, batch_first=True)
        
        # 线性层1，将LSTM的输出映射到128维
        self.fc1 = nn.Linear(64, 128)  # 将LSTM的输出维度128映射到64
        # 线性层2，将64维的输出映射到32维
        self.fc2 = nn.Linear(128, 128)   # 将64维映射到32
        # 线性层3，将32维的输出映射到最终输出的8维
        self.fc3 = nn.Linear(128, 64)    # 最终输出维度是8
        self.fc4 = nn.Linear(64, 8)    # 最终输出维度是8
        
    def forward(self, x):
        # x的形状是 (batch_size, seq_len, input_size)，例如 (256, 12, 14)
        
        # LSTM层的输出：lstm_out形状 (batch_size, seq_len, hidden_size)
        lstm_out, (hn, cn) = self.lstm(x)
        
        # 对LSTM输出的每个时间步都进行全连接层变换
        # 每个时间步的输出形状 (batch_size, seq_len, hidden_size)
        # 将LSTM的输出传入第一个全连接层
        x = torch.relu(self.fc1(lstm_out))  # 输出形状 (batch_size, seq_len, 64)
        
        # 经过第二个全连接层
        x = torch.relu(self.fc2(x))         # 输出形状 (batch_size, seq_len, 32)
        x = torch.relu(self.fc3(x))  
        # 经过第三个全连接层
        output = self.fc4(x)                # 输出形状 (batch_size, seq_len, 8)

        print(output.shape)  # 打印输出的形状
        return output
'''


def train_pinn(model, optimizer, data_loader, num_epochs= 20):
    batch_size = 256
    seq_length = 12

    # 用来存储每个 epoch 的损失
    epoch_losses = []

    for epoch in range(num_epochs):
        logging.info(f"epoch: {epoch}")
        total_loss = 0.0
        b = 0
        for batch in data_loader:
            logging.info(f"这是第{b}个batch")
            b += 1
            # 解包 batch 数据
            x_data, y_data = batch  # 数据获取
            # print(x_data.shape, y_data.shape)
            x_data = x_data.requires_grad_(True)  # 张量在反向传播的时候需要计算梯度
            y_pred = model(x_data)  # 预测扭矩和轨迹
                
            # 从预测输出中提取,前两个是轨迹,后六个是扭矩
            trajectory_pred = y_pred[:, :, :2]  # 前俩输出的是预测的轨迹
            # print("trajectory_pred", trajectory_pred.shape)
            tau_pred = y_pred[:, :, 2:]  # 后六个输出的是预测的扭矩
            # print("tau_pred", tau_pred.shape)
                
            # 真实的关节角度和关节速度
            q_data = x_data[:, :, 0:6]  # 关节角度
            dq_data = x_data[:, :, 6:12]  # 关节速度
            
            trajectory = y_data[:, :, 0:2]  # 真实轨迹
            # tau_data = y_data[:, :, 2:8]  # 关节扭矩
            dynamics_tau = y_data[:, :, 2:8]  # 动力学扭矩

            # # 扭矩的loss --- loss1   
            # data_loss = torch.mean((tau_pred - tau_data) ** 2)  # nn.MSELoss()(tau_pred, tau)
            # data_loss = data_loss.to(device)
            # logging.info(f"data_loss: {data_loss}")

            # 轨迹损失 --- loss2
            trajectory_loss = torch.mean((trajectory_pred - trajectory) ** 2)  # nn.MSELoss()(trajectory_pred, trajectory_true)
            trajectory_loss = trajectory_loss.to(device)
            logging.info(f"trajectory_loss: {trajectory_loss}")
            # print("trajectory_loss", trajectory_loss)

            # 动力学扭矩的loss --- loss3
            dynamics_loss = torch.mean((tau_pred - dynamics_tau) ** 2)  # nn.MSELoss()(dynamics_tau, tau_data)
            dynamics_loss = dynamics_loss.to(device)
            logging.info(f"dynamics_loss: {dynamics_loss}")

            # 总损失
            loss = 0.6*trajectory_loss + 0.4*dynamics_loss
            loss = loss.to(device)
            loss = loss.to(device)
            logging.info(f"loss: {loss}")
                
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                
            total_loss += loss.item()
            # total_loss = total_loss.to(device)
            logging.info(f"total_loss: {total_loss}")
            
        # 每个 epoch 打印平均损失
        avg_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
        logging.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")

# 训练主函数
def main():
    # 加载数据
    input_data, output_data = load_data_train('training_pusht_set26.txt')
    logging.info(f"开始加载训练数据")
    
    # 转换为 PyTorch 张量
    x_data = torch.tensor(input_data, dtype=torch.float32).to(device)  # q,qd,a，14维
    y_data = torch.tensor(output_data, dtype=torch.float32).to(device)  # a，2维
 
    # 创建数据集和数据加载器
    dataset = torch.utils.data.TensorDataset(x_data, y_data)
    # data_loader用于将数据集分成小批量，不随机打乱数据
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False, drop_last=True)  # pin_memory=True,
    print("data_loader", data_loader)
    
    # 初始化模型和优化器
    model = PINN_LSTM().to(device)  # 输入6+6+2,输出6+2
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # 训练模型
    num_epochs = 20  # 根据需要调整 epoch 数,就是遍历多少次数据集
    train_pinn(model, optimizer, data_loader, num_epochs=num_epochs)
    
    # 保存模型
    torch.save(model.state_dict(), 'pusht_LSTM_4_mask_model-250220-20.pth')


if __name__ == "__main__":
    main()
