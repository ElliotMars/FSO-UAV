import numpy as np
from filterpy.kalman import KalmanFilter

def kalman_filter(data):
    # 初始化卡尔曼滤波器
    kf = KalmanFilter(dim_x=4, dim_z=2)

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
    print(kf.P)

    # 定义过程噪声协方差
    kf.Q = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])

    # 定义测量噪声协方差
    kf.R = np.array([[1, 0],
                     [0, 1]])

    # 进行卡尔曼滤波
    filtered_coords = []
    for coord, flagf in data:
        if flagf:  # 只处理有效的坐标
            kf.predict()
            kf.update(coord)
            filtered_coords.append(kf.x[:2])
        else:
            filtered_coords.append(None)

    return np.array(filtered_coords, dtype=object)

# 示例数据序列（包含坐标和有效性标志）
data = [((1, 2), True), ((2, 3), False), ((3, 4), True), ((4, 5), True), ((5, 6), False)]

# 应用卡尔曼滤波
filtered_data = kalman_filter(data)
print(filtered_data)
