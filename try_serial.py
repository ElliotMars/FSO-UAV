import serial
import time

# 打开串口
ser = serial.Serial('COM6', 1000000, timeout=1)

# 注意：在 pySerial 中没有直接方法来设置 QueueSize
# 我们只能设置波特率和其他参数，队列大小可能需要通过操作系统或其他工具设置

# 设置串口参数
ser.bytesize = serial.EIGHTBITS  # 设置数据位为 8 位
ser.parity = serial.PARITY_NONE  # 设置校验位为无
ser.stopbits = serial.STOPBITS_ONE  # 设置停止位为 1 位
ser.xonxoff = False  # 禁用软件流控
ser.rtscts = False  # 禁用硬件流控 (RTS/CTS)
ser.dsrdtr = False  # 禁用硬件流控 (DSR/DTR)

# 十六进制数据字符串
data_str = 'A8 E5 48 00 01 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 \
00 00 00 01 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 \
00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 03 18 D1'
data_bytes = bytes.fromhex(data_str.replace(' ', ''))

# 发送数据
ser.write(data_bytes)

# 等待数据发送完毕
ser.flush()

# 关闭串口
ser.close()