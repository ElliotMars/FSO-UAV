#项目介绍
    本项目的最终效果是通过Yolo实现地对空的无人机识别、坐标获取，并通过强化学习实现激光通信。

#环境依赖

#目录结构描述

#使用说明
     Version-1.X使用说明：
1.运行hostname.py，输出本机IP。
2.将server.py第6行和detect.py第54行的IP改为本机IP。（此处可以优化）
3.先运行server.py，看到输出waiting后运行detect.py。（二者同时运行，但是需要server.py先启动）
如果遇到ConnectionRefusedError: [WinError 10061] 由于目标计算机积极拒绝，无法连接
可以参考https://blog.csdn.net/m0_52939861/article/details/117709011?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522169902604016800180672530%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=169902604016800180672530&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~rank_v31_ecpm-2-117709011-null-null.142^v96^pc_search_result_base1&utm_term=socket%20ConnectionRefusedError%3A%20%5BWinError%2010061%5D%20%E7%94%B1%E4%BA%8E%E7%9B%AE%E6%A0%87%E8%AE%A1%E7%AE%97%E6%9C%BA%E7%A7%AF%E6%9E%81%E6%8B%92%E7%BB%9D%EF%BC%8C%E6%97%A0%E6%B3%95%E8%BF%9E%E6%8E%A5&spm=1018.2226.3001.4187

#版本内容更新
######Version-1.0.0
      1.完成Yolov5模型训练。
      2.通过Socket进行坐标传输。
