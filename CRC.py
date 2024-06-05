from binascii import unhexlify
from crcmod import mkCrcFun



# CRC16/XMODEM

def crc16_xmodem(s):
    crc16 = mkCrcFun(0x11021, rev=False, initCrc=0x0000, xorOut=0x0000)
    return get_crc_value(s, crc16)

# common func
def get_crc_value(s, crc16):
    data = s.replace(' ', '')
    crc_out = crc16(unhexlify(data))
    crc_data = f"{crc_out:04X}"  # 转换为大写的4位十六进制字符串，自动补零
    return crc_data[:2] + ' ' + crc_data[2:]


if __name__ == '__main__':

    s4 = crc16_xmodem(f'A8E54800010000CEFF71020400000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000')

    print('crc16_xmodem: ' + s4)

