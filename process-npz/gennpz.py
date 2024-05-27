import numpy as np
import struct
import argparse

def hex_to_float(hex_str):
    """将十六进制字符串转换为32位浮点数"""
    return struct.unpack('!f', bytes.fromhex(hex_str))[0]

def read_hex_file(file_path):
    """读取包含十六进制数的文本文件，并转换为浮点数列表"""
    float_numbers = []
    with open(file_path, 'r') as file:
        for line in file:
            hex_values = line.strip().split()
            for hex_value in hex_values:
                # 移除 "0x" 前缀并转换为浮点数
                float_numbers.append(hex_to_float(hex_value[2:]))
    if len(float_numbers) != 1024:
        raise ValueError("文件中不包含正确数量的十六进制数，应该是1024个。")
    return np.array(float_numbers,dtype=np.float32)

def save_to_npz(float_array, output_path):
    """将浮点数数组保存为 .npz 文件，每512个数为一个数组"""
    if len(float_array) != 1024:
        raise ValueError("浮点数数组的长度不是1024。")
    array1 = float_array[:512]
    array2 = float_array[512:]
    np.savez(output_path, encoder_out=array1, decoder_out=array2)

def main():
    parser = argparse.ArgumentParser(description='读取十六进制32位浮点数并保存为npz文件')
    parser.add_argument('input_file', type=str, help='输入的包含十六进制数的文本文件')
    parser.add_argument('output_file', type=str, help='输出的npz文件')
    
    args = parser.parse_args()
    
    # 读取文件并转换为浮点数数组
    float_array = read_hex_file(args.input_file)
    
    # 保存为 .npz 文件
    save_to_npz(float_array, args.output_file)
    
    # 打印已保存的数据验证
    loaded_data = np.load(args.output_file)
    print("encoder_out:", loaded_data['encoder_out'])
    print("decoder_out:", loaded_data['decoder_out'])

if __name__ == '__main__':
    main()

