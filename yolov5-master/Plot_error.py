import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os


class ErrorPlot:
    def __init__(self, save_path='error_plots'):
        self.error_memo = []
        self.t = 1
        self.save_path = save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.fig, self.ax = plt.subplots()

    def add_error(self, error):
        self.error_memo.append(error)
        self.t += 1
        self.plotout()

    def plotout(self):
        if not self.error_memo:
            raise ValueError("没有错误可以绘制。请先使用 add_error() 方法添加错误。")

        t_values = list(range(1, self.t))
        if (len(self.error_memo)==20):
            self.ax.clear()
            self.ax.plot(t_values, self.error_memo, marker='o', linestyle='-', color='b', label='Error')
            self.ax.set_xlabel('t')
            self.ax.set_ylabel('Error')
            self.ax.set_title('Error Plot')
            self.ax.legend()
            self.ax.grid(True)
            # 保存当前帧
            file_path = os.path.join(self.save_path, f'frame_{self.t - 1}.png')
            plt.savefig('error-t')
            plt.pause(0.1)  # 暂停以更新图像
        else:
            pass

    def animate(self):
        ani = animation.FuncAnimation(self.fig, self.plotout, frames=len(self.error_memo), repeat=False)
        plt.show()
