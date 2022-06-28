#-*- coding: utf-8 -*-
import os
import numpy as np
import cv2
import shutil


def main(root_dir):
    files = ['WFLW/train_data/list.txt', '300VW_Dataset_2015_12_14/train_data/list.txt']
    dst_dir = os.path.join(root_dir, 'TOTAL_FACE/train_data')
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    dst_path = os.path.join(dst_dir, 'list.txt')
    
    with open(dst_path, 'w') as fw:
        for file_path in files:
            fp = os.path.join(root_dir, file_path)
            if os.path.isfile(fp):
                with open(fp, 'r') as fr:
                    lines = fr.readlines()
                    for index, line in enumerate(lines):
                        fw.write(line)
    
    test_files = ['WFLW/test_data/list.txt']
    test_dst_dir = os.path.join(root_dir, 'TOTAL_FACE/test_data')
    if not os.path.exists(test_dst_dir):
        os.makedirs(test_dst_dir)
    test_dst_path = os.path.join(test_dst_dir, 'list.txt')
    with open(test_dst_path, 'w') as fw:
        for file_path in test_files:
            fp = os.path.join(root_dir, file_path)
            if os.path.isfile(fp):
                with open(fp, 'r') as fr:
                    lines = fr.readlines()
                    for index, line in enumerate(lines):
                        fw.write(line)


if __name__ == '__main__':
    root_dir = "/data/Datasets/"

    print(root_dir)
    main(root_dir)