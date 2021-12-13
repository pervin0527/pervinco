import os
import xml.etree.ElementTree as ET
import pandas as pd
import cv2
import json

def write_to_xml(image_name, image_dict, data_folder, save_folder, xml_template='pascal_voc_template.xml'):
    bboxes = image_dict[image_name]
    
    tree = ET.parse(xml_template)
    root = tree.getroot()    
    
    folder = root.find('folder')
    folder.text = 'Annotations'
    
    fname = root.find('filename')
    fname.text = image_name.split('.')[0] 
    
    src = root.find('source')
    database = src.find('database')
    database.text = 'COCO2017'
    
    
    img = cv2.imread(os.path.join(data_folder, image_name))
    h,w,d = img.shape
    
    size = root.find('size')
    width = size.find('width')
    width.text = str(w)
    height = size.find('height')
    height.text = str(h)
    depth = size.find('depth')
    depth.text = str(d)
    
    for box in bboxes:

        obj = ET.SubElement(root, 'object')
        
        name = ET.SubElement(obj, 'name')
        name.text = box[0]
        
        pose = ET.SubElement(obj, 'pose')
        pose.text = 'Unspecified'

        truncated = ET.SubElement(obj, 'truncated')
        truncated.text = str(0)

        difficult = ET.SubElement(obj, 'difficult')
        difficult.text = str(0)

        bndbox = ET.SubElement(obj, 'bndbox')
        
        xmin = ET.SubElement(bndbox, 'xmin')
        xmin.text = str(int(box[1]))
        
        ymin = ET.SubElement(bndbox, 'ymin')
        ymin.text = str(int(box[2]))
        
        xmax = ET.SubElement(bndbox, 'xmax')
        xmax.text = str(int(box[3]))
        
        ymax = ET.SubElement(bndbox, 'ymax')
        ymax.text = str(int(box[4]))
    
    anno_path = os.path.join(save_folder, image_name.split('.')[0] + '.xml')
    print(anno_path)
    tree.write(anno_path)
    
if __name__=='__main__':
    annotations_path = 'annotations/instances_train2017.json'
    
    df = pd.read_csv('coco_categories.csv')
    df.set_index('id', inplace=True)
    
    image_folder = 'train2017'
    savepath = 'saved'
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    
    with open(annotations_path,'rb') as file:
        doc = json.load(file)
        
    annotations = doc['annotations']
    iscrowd_allowed = 1
    image_dict = {}
    
    for anno in annotations:

        image_id = anno['image_id']
        image_name = '{0:012d}.jpg'.format(image_id)    
        
        category = df.loc[anno['category_id']]['name']
        
        if not image_name in image_dict.keys():
            image_dict[image_name]=[]
        
        box = anno['bbox']

        image_dict[image_name].append([category, box[0], box[1], box[0]+box[2], box[1]+box[3]])
        
    for image_name in image_dict.keys():
        write_to_xml(image_name, image_dict, image_folder, savepath)
        print('generated for: ', image_name)