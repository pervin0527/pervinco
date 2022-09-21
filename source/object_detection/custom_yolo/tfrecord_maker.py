import hashlib
import lxml.etree
import pandas as pd
import tensorflow as tf

from glob import glob
from tqdm import tqdm


def read_label_file(path):
    df = pd.read_csv(path, sep=",", index_col=False, header=None)
    classes = df[0].to_list()

    return classes


def parse_xml(xml):
    if not len(xml):
        return {xml.tag: xml.text} 

    result = {}
    for child in xml:
        child_result = parse_xml(child)
        if child.tag != "object":
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    
    return {xml.tag: result}


def build_example(img_path, annotation, class_map):
    img_raw = open(img_path, 'rb').read()
    key = hashlib.sha256(img_raw).hexdigest()

    width = int(annotation['size']['width'])
    height = int(annotation['size']['height'])

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    truncated = []
    views = []
    difficult_obj = []
    if 'object' in annotation:
        for obj in annotation['object']:
            difficult = bool(int(obj['difficult']))
            difficult_obj.append(int(difficult))

            xmin.append(float(obj['bndbox']['xmin']) / width)
            ymin.append(float(obj['bndbox']['ymin']) / height)
            xmax.append(float(obj['bndbox']['xmax']) / width)
            ymax.append(float(obj['bndbox']['ymax']) / height)
            classes_text.append(obj['name'].encode('utf8'))
            classes.append(class_map[obj['name']])
            truncated.append(int(obj['truncated']))
            views.append(obj['pose'].encode('utf8'))

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[
            annotation['filename'].encode('utf8')])),
        'image/source_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[
            annotation['filename'].encode('utf8')])),
        'image/key/sha256': tf.train.Feature(bytes_list=tf.train.BytesList(value=[key.encode('utf8')])),
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
        'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=['jpeg'.encode('utf8')])),
        'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmin)),
        'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmax)),
        'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymin)),
        'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymax)),
        'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=classes_text)),
        'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=classes)),
        'image/object/difficult': tf.train.Feature(int64_list=tf.train.Int64List(value=difficult_obj)),
        'image/object/truncated': tf.train.Feature(int64_list=tf.train.Int64List(value=truncated)),
        'image/object/view': tf.train.Feature(bytes_list=tf.train.BytesList(value=views)),
    }))
    return example


def write_tfrecord():
    for s in split:
        writer = tf.io.TFRecordWriter(f"{data_dir}/{s}/{s}.tfrecord")
        image_list = sorted(glob(f"{data_dir}/{s}/images/*.jpg"))

        for idx in tqdm(range(len(image_list))):
            img_path = image_list[idx]
            file_name = img_path.split('/')[-1].split('.')[0]
            xml_file = f"{data_dir}/{s}/annotations/{file_name}.xml"
            xml_data = lxml.etree.fromstring(open(xml_file).read())
            annotation = parse_xml(xml_data)["annotation"]
            
            tf_example = build_example(img_path, annotation, classes)
            writer.write(tf_example.SerializeToString())
        
        writer.close()


if __name__ == "__main__":
    data_dir = "/data/Datasets/SPC/full-name14"
    label_dir = "/data/Datasets/SPC/Labels/labels.txt"
    split = ["train3", "valid3"]

    classes = {name: idx for idx, name in enumerate(open(label_dir).read().splitlines())}
    write_tfrecord()