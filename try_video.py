import cv2

# RTSP流的URL
rtsp_url = 'rtsp://192.168.144.108:554/sub'
print('!!!')
# 创建一个VideoCapture对象
cap = cv2.VideoCapture(rtsp_url)
print('aaa')

# 检查是否成功打开视频流
if not cap.isOpened():
    print("无法打开RTSP视频流")
    exit()

try:
    while True:
        # 读取视频流的下一帧
        ret, frame = cap.read()

        # 如果成功获取帧，显示它
        if ret:
            cv2.imshow('RTSP Stream', frame)

            # 按 'q' 退出循环
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("无法读取视频流的下一帧")
            break

finally:
    # 释放VideoCapture对象
    cap.release()
    cv2.destroyAllWindows()