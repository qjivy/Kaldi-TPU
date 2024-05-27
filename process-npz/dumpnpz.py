import numpy as np
import argparse

def read_and_print_npz(file_path):
    """读取 npz 文件并打印其中的内容"""
    data = np.load(file_path)
    for key, value in data.items():
        print(f'{key}: {value}')

def main():
    parser = argparse.ArgumentParser(description='读取 npz 文件并打印内容')
    parser.add_argument('input_file', type=str, help='输入的 npz 文件')
    
    args = parser.parse_args()
    
    read_and_print_npz(args.input_file)

if __name__ == '__main__':
    main()

