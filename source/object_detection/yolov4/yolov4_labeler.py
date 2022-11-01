from ctypes import *
import os
import cv2
import darknet

from tqdm import tqdm
from glob import glob
from xml.etree.ElementTree import ElementTree
from lxml.etree import Element, SubElement

def rewrite_xml(dst, results, idx):
    node_root = Element('annotation')
    
    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'folder'
    
    node_filename = SubElement(node_root, 'filename')
    node_filename.text = f'image_{idx}.jpg'
    
    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = str(width)
    
    node_height = SubElement(node_size, 'height')
    node_height.text = str(height)
    
    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '3'
    
    for i in range(len(results)):
        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text = str(results[i][0])

        node_pose = SubElement(node_object, 'pose')
        node_pose.text = 'Unspecified'

        node_truncated = SubElement(node_object, 'truncated')
        node_truncated.text = '0'

        node_occluded = SubElement(node_object, 'occluded')
        node_occluded.text = '0'

        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = '0'

        node_bndbox = SubElement(node_object, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = str(results[i][1])
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = str(results[i][2])
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = str(results[i][3])
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = str(results[i][4])
        
    tree = ElementTree(node_root)
    tree.write(dst)


def save_detection_result(path):
    folders = sorted(glob(f"{path}/*"))

    for folder in folders:
        folder_name = folder.split('/')[-1]
        if not os.path.isdir(f"{output_path}/{folder_name}"):
            os.makedirs(f"{output_path}/{folder_name}/JPEGImages")
            os.makedirs(f"{output_path}/{folder_name}/Annotations")
            os.makedirs(f"{output_path}/{folder_name}/Results")

        images = sorted(glob(f"{folder}/*.jpg"))
        for idx in tqdm(range(len(images))):
            test_image = cv2.imread(images[idx])
            frame_rgb = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (width, height))
            darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())

            detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh_hold, hier_thresh=.5, nms=.45)
            just_show = frame_resized.copy()
            image = darknet.draw_boxes(detections, just_show, class_colors)

            if detections and float(detections[0][1]) > thresh_hold:
                result = []
                for i in range(len(detections)):
                    class_name = detections[i][0]
                    x = detections[i][2][0]
                    y = detections[i][2][1]
                    w = detections[i][2][2]
                    h = detections[i][2][3]

                    xmin, ymin, xmax, ymax = darknet.bbox2points((x, y, w, h))
                    result.append([class_name, xmin, ymin, xmax, ymax])

                frame_resized = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                cv2.imwrite(f"{output_path}/{folder_name}/JPEGImages/{idx:>09}.jpg", frame_resized)
                cv2.imwrite(f"{output_path}/{folder_name}/Results/{idx:>09}.jpg", image)
                rewrite_xml(f'{output_path}/{folder_name}/Annotations/{idx:>09}.xml', result, idx)


if __name__ == "__main__":
    weight_file = "/home/ubuntu/Models/BR/yolov4_last.weights"
    config_file = "/home/ubuntu/darknet/custom/yolov4.cfg"
    data_file = "/home/ubuntu/darknet/custom/obj.data"

    image_path = "/home/ubuntu/Datasets/BR/frames"
    output_path = "/home/ubuntu/Datasets/BR/semi-label"
    thresh_hold = .9

    network, class_names, class_colors = darknet.load_network(config_file, data_file, weight_file, batch_size=1)
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)

    save_detection_result(image_path)