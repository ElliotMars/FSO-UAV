import cv2
import threading

class VideoStreamWidget:
    def __init__(self, src=0):
        # 打开视频流
        self.capture = cv2.VideoCapture(src)
        if not self.capture.isOpened():
            print("Error: Unable to open video source.")
        
        # 读取第一帧
        self.status, self.frame = self.capture.read()
        
        # 启动一个线程来不断读取视频帧
        threading.Thread(target=self.update, args=()).start()

    def update(self):
        # 不断读取视频流中的帧
        while True:
            if self.capture.isOpened():
                self.status, self.frame = self.capture.read()

    def show_frame(self):
        # 显示当前帧
        if self.frame is not None:
            cv2.imshow('frame', self.frame)
            # 按下 'q' 键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.capture.release()
                cv2.destroyAllWindows()

# 使用示例
stream_link = 'rtsp://192.168.144.108:554'  # 实际的RTSP视频流链接
video_stream_widget = VideoStreamWidget(src=stream_link)

while True:
    try:
        video_stream_widget.show_frame()
    except AttributeError:
        pass
    except Exception as e:
        print(f"An error occurred: {e}")
