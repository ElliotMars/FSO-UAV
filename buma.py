def float_to_hex_complement(value):
    # 1. 取两位小数并乘以100
    scaled_value = round(value * 100, 2)

    # 2. 转换为整数
    int_value = int(scaled_value)

    # 3. 获取补码（16位整数表示）
    if int_value < 0:
        complement = (1 << 16) + int_value
    else:
        complement = int_value

    # 4. 转换为16进制表示，确保为4位
    hex_value = format(complement & 0xFFFF, '04x')  # 取16位，并确保长度为4

    # 5. 小端序（交换字节顺序）
    little_endian_hex = hex_value[2:] + hex_value[:2]

    # 6. 将字母大写
    little_endian_hex = little_endian_hex.upper()

    return little_endian_hex

print(float_to_hex_complement(-10))