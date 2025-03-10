import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import logging
import datetime
import matplotlib

# 设置Matplotlib使用支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号
# 获取当前日期并格式化
current_date = datetime.datetime.now().strftime("%Y-%m-%d")
log_filename = f"PINN-Test-{current_date}-pusht-LSTM_4-10.log"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用的设备: {device}")

# 配置基本日志设置
logging.basicConfig(
    level=logging.DEBUG,  # 设置最低日志级别
    format='%(asctime)s - %(levelname)s - %(message)s',  # 日志格式
    datefmt='%Y-%m-%d %H:%M:%S',  # 时间格式
    filename=log_filename,  # 日志文件名
    filemode='w'  # 文件写入模式，'w' 表示覆盖，'a' 表示追加
)

# 如果需要同时输出到控制台，可以添加一个 StreamHandler
console = logging.StreamHandler()
console.setLevel(logging.INFO)  # 设置控制台日志级别
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

# 数据读取和预处理
def load_data_test_full(filename):
    """
    加载测试数据，假设数据文件包含关节角度（6）、关节速度（6）和末端执行器位置（2），共14维。
    """
    # 读取数据
    data = np.loadtxt(filename)
    
    # 数据拆分
    # joint_angles = data[:, 0:6]      # q1 - q6
    # joint_velocities = data[:, 6:12] # dq1 - dq6
    joint_angles = np.zeros_like(data[:, 0:6])      # 6 个关节角度 = 0
    joint_velocities = np.zeros_like(data[:, 6:12]) # 6 个关节速度 = 0
    # tau = data[:, 12:18] # tau1 - tau6
    end_effector_positions = data[:, 18:20] # x_end, y_end
    dynamics_tau = data[:, 20:26] # tau1 - tau6

    # 构建序列数据
    sequence_length = 12
    num_sequences = int((joint_angles.shape[0] - sequence_length) / 1) + 1
    
    # 初始化数据列表
    input_data = []
    output_data = []
    
    for i in range(num_sequences):
        # 输入特征：关节角度、关节速度、末端位置
        s = i
        input_seq = np.hstack([
            joint_angles[s:s+sequence_length],
            joint_velocities[s:s+sequence_length],
            end_effector_positions[s:s+sequence_length]
        ])
        # 输出特征：末端位置
        # 输出特征
        output_seq = np.hstack([
            end_effector_positions[s:s+sequence_length],
            # tau[s:s+sequence_length],
            dynamics_tau[s:s+sequence_length]
        ])

        input_data.append(input_seq)
        output_data.append(output_seq)
    
    # 转换为 NumPy 数组
    input_data = np.array(input_data)
    output_data = np.array(output_data)
    print(input_data.shape, output_data.shape)

    return input_data, output_data

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

def evaluate_model(model, data_loader):
    model.eval()  # 设置模型为评估模式
    total_loss = 0.0
    criterion = nn.MSELoss()
    predictions = []
    ground_truth = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            x_data, y_data = batch
            # x_data[:, :, :12] = 0  # 将前 12 维置零
            print(x_data.shape)
            print(x_data.type)
            x_data = x_data.to(device)
            y_data = y_data.to(device)
            
            outputs = model(x_data)  #!!!!!!!!!!!!!!!!
            loss = criterion(outputs, y_data)
            total_loss += loss.item()
            
            predictions.append(outputs.cpu().numpy())
            ground_truth.append(y_data.cpu().numpy())
            
            if batch_idx % 10 == 0:
                logging.info(f"处理第 {batch_idx} 个批次，当前损失: {loss.item()}")
    
    avg_loss = total_loss / len(data_loader)
    logging.info(f"测试集平均损失 (MSE): {avg_loss:.6f}")
    print(f"测试集平均损失 (MSE): {avg_loss:.6f}")
    
    # 将列表转换为 NumPy 数组
    predictions = np.concatenate(predictions, axis=0)
    ground_truth = np.concatenate(ground_truth, axis=0)
    
    return predictions, ground_truth, avg_loss



def compute_smoothness_metrics(trajectory):
    """
    计算轨迹的平滑度，包括：
    - 总曲率变化量（TCV）
    - 加速度变化量（AV）
    :param trajectory: 形状为 (N, 2) 的 NumPy 数组，表示 x, y 轨迹
    :return: TCV, AV
    """
    # 计算一阶速度 (dx/dt, dy/dt)
    velocity = np.diff(trajectory, axis=0)  # 计算相邻点之间的差分
    speed = np.linalg.norm(velocity, axis=1) + 1e-6  # 避免除零

    # 计算二阶加速度 (d^2x/dt^2, d^2y/dt^2)
    acceleration = np.diff(velocity, axis=0)  # 再次计算相邻点之间的差分
    acc_magnitude = np.linalg.norm(acceleration, axis=1)  # 计算加速度模长

    # 计算曲率 k = |dx/dt * d²y/dt² - dy/dt * d²x/dt²| / (dx/dt² + dy/dt²)^(3/2)
    curvature = np.abs(
        velocity[:-1, 0] * acceleration[:, 1] - velocity[:-1, 1] * acceleration[:, 0]
    ) / (speed[:-1] ** 3)

    # 计算总曲率变化量 (TCV)
    tcv = np.sum(np.abs(np.diff(curvature)))

    # 计算加速度变化量 (AV)
    av = np.sum(np.abs(np.diff(acc_magnitude)))

    return tcv, av

def main():
    # 加载测试数据
    test_filename = 'testing_pusht_set26.txt'  
    logging.info("开始加载测试数据")
    input_data, output_data = load_data_test_full(test_filename)
    logging.info(f"测试数据加载完成: 输入形状 {input_data.shape}, 输出形状 {output_data.shape}")
    
    # 转换为 PyTorch 张量
    x_test = torch.tensor(input_data, dtype=torch.float32).to(device)
    y_test = torch.tensor(output_data, dtype=torch.float32).to(device)
    
    # 创建数据集和数据加载器
    dataset = torch.utils.data.TensorDataset(x_test, y_test)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False, drop_last=False)
    
    logging.info("测试数据加载器创建完成")
    
    # 初始化模型并加载训练好的权重
    model = PINN_LSTM().to(device)
    model_path = 'pusht_LSTM_4_mask_model-250220-10.pth'  
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  
    logging.info(f"模型已加载: {model_path}")
    
    # 评估模型
    predictions, ground_truth, avg_loss = evaluate_model(model, data_loader)
    
    # 计算轨迹的平滑度
    tcv_gt_list, av_gt_list = [], []
    tcv_pred_list, av_pred_list = [], []

    for i in range(ground_truth.shape[0]):
        gt_trajectory = ground_truth[i, 0:7, :2]  # 提取 ground truth 轨迹 (x, y)
        pred_trajectory = predictions[i, 0:7, :2]  # 提取预测轨迹 (x, y)
        
        tcv_gt, av_gt = compute_smoothness_metrics(gt_trajectory)
        tcv_pred, av_pred = compute_smoothness_metrics(pred_trajectory)

        tcv_gt_list.append(tcv_gt)
        av_gt_list.append(av_gt)
        tcv_pred_list.append(tcv_pred)
        av_pred_list.append(av_pred)

    # 计算总体平滑度均值
    avg_tcv_gt, avg_av_gt = np.mean(tcv_gt_list), np.mean(av_gt_list)
    avg_tcv_pred, avg_av_pred = np.mean(tcv_pred_list), np.mean(av_pred_list)

    logging.info(f"真实轨迹的总曲率变化量 (TCV): {avg_tcv_gt:.6f}, 加速度变化量 (AV): {avg_av_gt:.6f}")
    logging.info(f"预测轨迹的总曲率变化量 (TCV): {avg_tcv_pred:.6f}, 加速度变化量 (AV): {avg_av_pred:.6f}")
    print(f"真实轨迹的总曲率变化量 (TCV): {avg_tcv_gt:.6f}, 加速度变化量 (AV): {avg_av_gt:.6f}")
    print(f"预测轨迹的总曲率变化量 (TCV): {avg_tcv_pred:.6f}, 加速度变化量 (AV): {avg_av_pred:.6f}")

    # 绘制直方图对比平滑度
    plt.figure(figsize=(8, 6))
    plt.hist(tcv_gt_list, bins=30, alpha=0.5, label='真实轨迹 TCV', color='blue')
    plt.hist(tcv_pred_list, bins=30, alpha=0.5, label='预测轨迹 TCV', color='red')
    plt.xlabel("总曲率变化量 (TCV)")
    plt.ylabel("频数")
    plt.title("真实 vs. 预测轨迹 TCV 分布")
    plt.legend()
    plt.savefig("TCV_comparison.png")
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.hist(av_gt_list, bins=30, alpha=0.5, label='真实轨迹 AV', color='blue')
    plt.hist(av_pred_list, bins=30, alpha=0.5, label='预测轨迹 AV', color='red')
    plt.xlabel("加速度变化量 (AV)")
    plt.ylabel("频数")
    plt.title("真实 vs. 预测轨迹 AV 分布")
    plt.legend()
    plt.savefig("AV_comparison.png")
    plt.close()
    logging.info("保存平滑度对比直方图")

    # 可视化部分结果
    '''
    num_plots = 5  # 绘制5个样本(数据集中前五个序列)
    for i in range(num_plots):
        k = i  * 5  # 从k->k+12
        plt.figure(figsize=(8, 6))
        plt.plot(ground_truth[k, :, 0], ground_truth[k, :, 1], 'b-', label='真实轨迹')
        plt.plot(predictions[k, :, 0], predictions[k, :, 1], 'r-', label='预测轨迹')
        plt.title(f"样本 {i+1} 轨迹对比")
        plt.xlabel("x_end")
        plt.ylabel("y_end")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"trajectory_comparison_sample_{i+1}.png")
        plt.close()
        logging.info(f"保存样本 {i+1} 的轨迹对比图")
    '''

    num_plots = 5  # 绘制5个样本(数据集中前五个序列)
    for i in range(num_plots):
        k = i * 5  # 从k->k+12
        
        # 计算欧几里得误差
        errors = np.linalg.norm(ground_truth[k] - predictions[k], axis=1)
        rmse = np.sqrt(np.mean(errors**2))  # 计算均方根误差
        
        plt.figure(figsize=(8, 6))
        plt.plot(ground_truth[k, :, 0], ground_truth[k, :, 1], 'b-', label='真实轨迹')
        plt.plot(predictions[k, :, 0], predictions[k, :, 1], 'r-', label='预测轨迹')
        
        # 在图像上显示RMSE
        plt.text(0.05, 0.95, f"RMSE: {rmse:.4f}", transform=plt.gca().transAxes,
                fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
        
        plt.title(f"样本 {i+1} 轨迹对比")
        plt.xlabel("x_end")
        plt.ylabel("y_end")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"trajectory_comparison_sample_{i+1}.png")
        plt.close()
        
        logging.info(f"保存样本 {i+1} 的轨迹对比图，RMSE: {rmse:.4f}")


    # 计算整体的均方误差
    mse = np.mean((predictions - ground_truth) ** 2)
    logging.info(f"整体均方误差 (MSE): {mse:.6f}")
    print(f"整体均方误差 (MSE): {mse:.6f}")
    
    # 计算每个维度的MSE
    mse_x = np.mean((predictions[:, :, 0] - ground_truth[:, :, 0]) ** 2)
    mse_y = np.mean((predictions[:, :, 1] - ground_truth[:, :, 1]) ** 2)
    logging.info(f"x_end 的均方误差 (MSE): {mse_x:.6f}")
    logging.info(f"y_end 的均方误差 (MSE): {mse_y:.6f}")
    print(f"x_end 的均方误差 (MSE): {mse_x:.6f}")
    print(f"y_end 的均方误差 (MSE): {mse_y:.6f}")
    
    # 绘制总体预测与真实值的散点图
    plt.figure(figsize=(8, 6))
    plt.scatter(ground_truth[:, :, 0].flatten(), ground_truth[:, :, 1].flatten(),
                c='blue', alpha=0.5, label='真实轨迹')
    plt.scatter(predictions[:, :, 0].flatten(), predictions[:, :, 1].flatten(),
                c='red', alpha=0.5, label='预测轨迹')
    plt.title("总体轨迹散点图对比")
    plt.xlabel("x_end")
    plt.ylabel("y_end")
    plt.legend()
    plt.grid(True)
    plt.savefig("overall_trajectory_scatter_comparison.png")
    plt.close()
    logging.info("保存总体轨迹散点图对比图")
    
    # 如果需要，可以保存预测结果
    np.save('predictions.npy', predictions)
    np.save('ground_truth.npy', ground_truth)
    logging.info("预测结果和真实值已保存为 .npy 文件")

if __name__ == "__main__":
    main()



'''
def main():
    # 加载测试数据
    test_filename = 'testing_set26.txt'  # 替换为您的测试集文件名
    logging.info("开始加载测试数据")
    input_data, output_data = load_data_test_full(test_filename)
    logging.info(f"测试数据加载完成: 输入形状 {input_data.shape}, 输出形状 {output_data.shape}")
    
    # 转换为 PyTorch 张量
    x_test = torch.tensor(input_data, dtype=torch.float32).to(device)  # 输入特征
    y_test = torch.tensor(output_data, dtype=torch.float32).to(device)  # 真实轨迹
    
    # 创建数据集和数据加载器
    dataset = torch.utils.data.TensorDataset(x_test, y_test)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False, drop_last=False)
    logging.info("测试数据加载器创建完成")
    
    # 初始化模型并加载训练好的权重
    model = PINN_LSTM().to(device)
    model_path = 'LSTM_4_mask_model-250212-10.pth'  # 'MLP_4_model-250111-1.pth'  # 替换为您的模型文件名
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # 设置为评估模式
    logging.info(f"模型已加载: {model_path}")
    
    # 评估模型
    predictions, ground_truth, avg_loss = evaluate_model(model, data_loader)
    
    # 可视化部分结果
    num_plots = 5  # 绘制5个样本(数据集中前五个序列)
    for i in range(num_plots):
        k = i  * 5  # 从k->k+12
        plt.figure(figsize=(8, 6))
        plt.plot(ground_truth[k, :, 0], ground_truth[k, :, 1], 'b-', label='真实轨迹')
        plt.plot(predictions[k, :, 0], predictions[k, :, 1], 'r-', label='预测轨迹')
        plt.title(f"样本 {i+1} 轨迹对比")
        plt.xlabel("x_end")
        plt.ylabel("y_end")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"trajectory_comparison_sample_{i+1}.png")
        plt.close()
        logging.info(f"保存样本 {i+1} 的轨迹对比图")
    
    # 计算整体的均方误差
    mse = np.mean((predictions - ground_truth) ** 2)
    logging.info(f"整体均方误差 (MSE): {mse:.6f}")
    print(f"整体均方误差 (MSE): {mse:.6f}")
    
    # 计算每个维度的MSE
    mse_x = np.mean((predictions[:, :, 0] - ground_truth[:, :, 0]) ** 2)
    mse_y = np.mean((predictions[:, :, 1] - ground_truth[:, :, 1]) ** 2)
    logging.info(f"x_end 的均方误差 (MSE): {mse_x:.6f}")
    logging.info(f"y_end 的均方误差 (MSE): {mse_y:.6f}")
    print(f"x_end 的均方误差 (MSE): {mse_x:.6f}")
    print(f"y_end 的均方误差 (MSE): {mse_y:.6f}")
    
    # 绘制总体预测与真实值的散点图
    plt.figure(figsize=(8, 6))
    plt.scatter(ground_truth[:, :, 0].flatten(), ground_truth[:, :, 1].flatten(),
                c='blue', alpha=0.5, label='真实轨迹')
    plt.scatter(predictions[:, :, 0].flatten(), predictions[:, :, 1].flatten(),
                c='red', alpha=0.5, label='预测轨迹')
    plt.title("总体轨迹散点图对比")
    plt.xlabel("x_end")
    plt.ylabel("y_end")
    plt.legend()
    plt.grid(True)
    plt.savefig("overall_trajectory_scatter_comparison.png")
    plt.close()
    logging.info("保存总体轨迹散点图对比图")
    
    # 如果需要，可以保存预测结果
    np.save('predictions.npy', predictions)
    np.save('ground_truth.npy', ground_truth)
    logging.info("预测结果和真实值已保存为 .npy 文件")

if __name__ == "__main__":
    main()
'''