import sys
import os
import numpy as np
import xml.etree.ElementTree as ET

from sklearn.cluster import KMeans

def xml_to_boxes(path, rescale_width=None, rescale_height=None):
    xml_list = []
    filenames = os.listdir(os.path.join(path))
    filenames = [os.path.join(path, f) for f in filenames if (f.endswith('.xml'))]
    
    for xml_file in filenames:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            bndbox = member.find('bndbox')
            bbox_width = int(bndbox.find('xmax').text) - int(bndbox.find('xmin').text)
            bbox_height = int(bndbox.find('ymax').text) - int(bndbox.find('ymin').text)
            if rescale_width and rescale_height:
              size = root.find('size')
              bbox_width = bbox_width * (rescale_width / int(size.find('width').text))
              bbox_height = bbox_height * (rescale_height / int(size.find('height').text))
        xml_list.append([bbox_width, bbox_height])
    
    bboxes = np.array(xml_list)
    
    return bboxes


def average_iou(bboxes, anchors):
    intersection_width = np.minimum(anchors[:, [0]], bboxes[:, 0]).T
    intersection_height = np.minimum(anchors[:, [1]], bboxes[:, 1]).T

    if np.any(intersection_width == 0) or np.any(intersection_height == 0):
        raise ValueError("Some boxes have zero size.")

    intersection_area = intersection_width * intersection_height
    boxes_area = np.prod(bboxes, axis=1, keepdims=True)
    anchors_area = np.prod(anchors, axis=1, keepdims=True).T
    union_area = boxes_area + anchors_area - intersection_area
    avg_iou_perc = np.mean(np.max(intersection_area / union_area, axis=1)) * 100

    return avg_iou_perc

def kmeans_aspect_ratios(bboxes, kmeans_max_iter, num_aspect_ratios):
    normalized_bboxes = bboxes / np.sqrt(bboxes.prod(axis=1, keepdims=True))
  
    kmeans = KMeans(init='random', n_clusters=num_aspect_ratios, random_state=0, max_iter=kmeans_max_iter)
    kmeans.fit(X=normalized_bboxes)
    ar = kmeans.cluster_centers_
    avg_iou_perc = average_iou(normalized_bboxes, ar)

    if not np.isfinite(avg_iou_perc):
        sys.exit("Failed to get aspect ratios due to numerical errors in k-means")

    aspect_ratios = [w/h for w,h in ar]

    return aspect_ratios, avg_iou_perc


if __name__ == "__main__":
    annot_path = "/home/ubuntu/Datasets/BR/set0_384/train/Annotations"
    aspect_ratios = 5 ## can be [2, 3, 4, 5, 6]
    kmeans_max_iter = 100000
    height, width = 384, 384

    bboxes = xml_to_boxes(path=annot_path)

    aspect_ratios, avg_iou_perc =  kmeans_aspect_ratios(bboxes=bboxes,
                                                        kmeans_max_iter=kmeans_max_iter,
                                                        num_aspect_ratios=aspect_ratios)

    aspect_ratios = sorted(aspect_ratios)

    print('Aspect ratios generated:', [round(ar,2) for ar in aspect_ratios])
    print('Average IOU with anchors:', avg_iou_perc)