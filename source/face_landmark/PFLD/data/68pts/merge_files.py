#-*- coding: utf-8 -*-
import os
import shutil

def main(root_dir):
    files = ['WFLW/train_data_68pts/list.txt', 'WFLW/custom/train_data_68pts/list.txt']
    dst_dir = os.path.join(root_dir, 'CUSTOM_MERGE/train_data_68pts')
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
        os.makedirs(f"{dst_dir}/imgs")
    dst_path = os.path.join(dst_dir, 'list.txt')
    
    with open(dst_path, 'w') as fw:
        for file_path in files:
            fp = os.path.join(root_dir, file_path)
            if os.path.isfile(fp):
                with open(fp, 'r') as fr:
                    lines = fr.readlines()
                    for index, line in enumerate(lines):
                        image_path = line.split()[0]
                        image_file_name = image_path.split('/')[-1]
                        shutil.copyfile(f"{image_path}", f"{dst_dir}/imgs/{image_file_name}")
                        fw.write(line)
    
    # test_files = ['WFLW/test_data_68pts/list.txt']
    # test_dst_dir = os.path.join(root_dir, 'TOTAL_FACE/test_data_68pts')
    # if not os.path.exists(test_dst_dir):
    #     os.makedirs(test_dst_dir)
    #     os.makedirs(f"{test_dst_dir}/imgs")
    # test_dst_path = os.path.join(test_dst_dir, 'list.txt')

    # with open(test_dst_path, 'w') as fw:
    #     for file_path in test_files:
    #         fp = os.path.join(root_dir, file_path)
    #         if os.path.isfile(fp):
    #             with open(fp, 'r') as fr:
    #                 lines = fr.readlines()
    #                 for index, line in enumerate(lines):
    #                     image_path = line.split()[0]
    #                     image_file_name = image_path.split('/')[-1]
    #                     shutil.copyfile(f"{image_path}", f"{test_dst_dir}/imgs/{image_file_name}")
    #                     fw.write(line)
                        

if __name__ == '__main__':
    root_dir = "/data/Datasets/"

    print(root_dir)
    main(root_dir)