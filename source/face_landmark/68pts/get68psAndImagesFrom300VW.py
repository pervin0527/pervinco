import os
import math
import numpy as np
from pathlib import Path
import cv2

num_kpts = 68

def get_kpts(file_path):
    kpts = []
    with open(file_path, 'r') as fr:
        ln = fr.readline()
        while not ln.startswith('n_points'):
            ln = fr.readline()

        num_pts = ln.split(':')[1].strip()

        # checking for the number of keypoints
        if float(num_pts) != num_kpts:
            # print("encountered file with less than %f keypoints in %s" %(num_kpts, file_pts))
            return None

        # skipping the line with '{'
        ln = fr.readline()

        ln = fr.readline()
        while not ln.startswith('}'):
            vals = ln.strip().split(' ')[:2]
            vals = list(map(np.float32, vals))
            kpts.append(vals)
            ln = fr.readline()
    return kpts


def get_Infomation_list(root_dir, info_dir, lines):
    info_path = os.path.join(root_dir, info_dir)
    img_dir = os.path.join(info_dir + '/imgs')
    img_path = os.path.join(root_dir, img_dir)
    if not os.path.exists(img_path):
        os.mkdir(img_path)
    vc = cv2.VideoCapture(info_path + '/vid.avi')
    index = 1
    while True:
        ret, frame = vc.read()
        if not ret:
            break
        image_name = '%06d.jpg' % index
        cv2.imwrite(os.path.join(img_path, image_name), frame)
        pts_path = info_path + ('/annot/%06d.pts' % index)
        #print(pts_path)
        
        kpts = get_kpts(pts_path)
        if kpts is None:
            continue
        # ibug dir exist a image that has a space in name

        GT_points = np.asarray(kpts)
        # crop face box
        x_min, y_min = GT_points.min(0)
        x_max, y_max = GT_points.max(0)
        w, h = x_max-x_min, y_max-y_min
        w = h = min(w, h)
        ratio = 0.1
        x_new = x_min - w*ratio
        y_new = y_min - h*ratio
        w_new = w*(1 + 2*ratio)
        h_new = h*(1 + 2*ratio)
        x1 = x_new
        x2 = x_new+w_new
        y1 = y_new
        y2 = y_new+h_new

        line = []
        for i in range(num_kpts):
            line.append(str(kpts[i][0]))
            line.append(str(kpts[i][1]))
        line.append(str(int(x1)))
        line.append(str(int(y1)))
        line.append(str(int(x2)))
        line.append(str(int(y2)))
        for i in range(6):
            line.append(str(0))
        line.append(os.path.join(img_dir, image_name))
        assert(len(line) == 147)
        lines.append(line)
        index += 1


def main(root_dir, fw_path_train, fw_path_test):
    train_lines = []
    train_dirs = os.listdir(root_dir+'/original')
    for i, train_dir in enumerate(train_dirs):
        print('%d/%d,' % (i+1, len(train_dirs)), train_dir)
        get_Infomation_list(root_dir+'/original', train_dir, train_lines)
    
    # test_lines = []
    # test_dirs = ['lfpw/testset', 'helen/testset']
    # for test_dir in test_dirs:
    #     get_Infomation_list(root_dir+'/raw', test_dir, test_lines)

    with open(fw_path_train, 'w') as fw:
        for i, line in enumerate(train_lines):
            #print(line)
            for j in range(len(line)):
                fw.write(line[j]+' ')
            fw.write('\n')

    # with open(fw_path_test, 'w') as fw:
    #     for i, line in enumerate(test_lines):
    #         for j in range(len(line)):
    #             fw.write(line[j]+' ')
    #         fw.write('\n')

if __name__ == '__main__':
    root_dir = "/data/Datasets/300VW_Dataset_2015_12_14"
    print(root_dir)

    if not os.path.isdir(f"{root_dir}/annotations/list_68pt_rect_attr_train_test"):
        os.makedirs(f"{root_dir}/annotations/list_68pt_rect_attr_train_test")  

    fw_path_train = f'{root_dir}/annotations/list_68pt_rect_attr_train_test/list_68pt_rect_attr_train.txt'
    fw_path_test = f'{root_dir}/annotations/list_68pt_rect_attr_train_test/list_68pt_rect_attr_test.txt'
    main(root_dir, fw_path_train, fw_path_test)
