import numpy as np
from filterpy.kalman import KalmanFilter

def kalman_filter(data):
    # 初始化卡尔曼滤波器
    kf = KalmanFilter(dim_x=4, dim_z=2)

    #定义初始状态向量
    x_kminus1 = data[1][0]
    y_kminus1 = data[1][1]
    delta_x = data[1][0]-data[0][0]
    delta_y = data[1][1] - data[0][1]
    kf.x = np.array([x_kminus1,y_kminus1,delta_x,delta_y])

    # 定义状态转换矩阵
    kf.F = np.array([[1, 0, 1, 0],
                     [0, 1, 0, 1],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])

    # 定义测量函数
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])

    # 定义状态协方差矩阵
    kf.P *= 1

    # 定义过程噪声协方差
    kf.Q = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])

    # 定义测量噪声协方差
    kf.R = np.array([[1, 0],
                     [0, 1]])

    # 进行卡尔曼滤波
    kf.predict()
    kf.update(data[2])
    filtered_coords = kf.x
    return np.array(filtered_coords, dtype=object)

# 示例数据序列（包含坐标和有效性标志）
#data = [(4,8),(1,4),(5,8)]

# 应用卡尔曼滤波
#filtered_data = kalman_filter(data)
#print(filtered_data,filtered_data[0],filtered_data[1])
