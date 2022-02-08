import darknet
import xml.etree.ElementTree as ET
from lxml.etree import Element, SubElement, tostring

def output_remake(detect_result):
    labels, scores, bboxes = [], [], []
    for dr in detect_result:
        labels.append(dr[0])
        scores.append(float(dr[1]))
        bboxes.append((darknet.bbox2points(dr[2])))

    return labels, scores, bboxes

def write_xml(save_dir, bboxes, labels, img_name, img_shape):
    root = Element("annotation")
    
    folder = SubElement(root, "folder")
    folder.text = "images"

    file_name = SubElement(root, "filename")
    file_name.text = f"{img_name}.jpg"

    size = SubElement(root, "size")
    w = SubElement(size, "width")
    h = SubElement(size, "height")
    h.text = str(img_shape[0])
    w.text = str(img_shape[1])
    depth = SubElement(size, "depth")
    depth.text = "3"

    if labels:
        for label, bbox in zip(labels, bboxes):
            obj = SubElement(root, "object")
            name = SubElement(obj, "name")
            name.text = label
            pose = SubElement(obj, "pose")
            pose.text = "0"
            truncated = SubElement(obj, 'truncated')
            truncated.text = '0'
            difficult = SubElement(obj, 'difficult')
            difficult.text = '0'
            bndbox = SubElement(obj, 'bndbox')
            xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]

            node_xmin = SubElement(bndbox, 'xmin')
            node_xmin.text = str(int(xmin))
            node_ymin = SubElement(bndbox, 'ymin')
            node_ymin.text = str(int(ymin))
            node_xmax = SubElement(bndbox, 'xmax')
            node_xmax.text = str(int(xmax))
            node_ymax = SubElement(bndbox, 'ymax')
            node_ymax.text = str(int(ymax))
    
    tree = ET.ElementTree(root)    
    tree.write(f"{save_dir}/{img_name}.xml")