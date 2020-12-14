# -*- coding: utf-8 -*-

import xml.etree.ElementTree as ET
from xml.dom import minidom
import os
import glob

lut={}
lut["airplane"] = 0
lut["apple"] = 1
lut["backpack"] = 2
lut["banana"] = 3
lut["baseball bat"] = 4
lut["baseball glove"] = 5
lut["bear"] = 6
lut["bed"] = 7
lut["bench"] = 8
lut["bicycle"] = 9
lut["bird"] = 10
lut["boat"] = 11
lut["book"] = 12
lut["bottle"] = 13
lut["bowl"] = 14
lut["broccoli"] = 15
lut["bus"] = 16
lut["cake"] = 17
lut["car"] = 18
lut["carrot"] = 19
lut["cat"] = 20
lut["cell phone"] = 21
lut["chair"] = 22
lut["clock"] = 23
lut["couch"] = 24
lut["cow"] = 25
lut["cup"] = 26
lut["dining table"] = 27
lut["dog"] = 28
lut["donut"] = 29
lut["elephant"] = 30
lut["fire hydrant"] = 31
lut["fork"] = 32
lut["frisbee"] = 33
lut["giraffe"] = 34
lut["hair drier"] = 35
lut["handbag"] = 36
lut["horse"] = 37
lut["hot dog"] = 38
lut["keyboard"] = 39
lut["kite"] = 40
lut["knife"] = 41
lut["laptop"] = 42
lut["microwave"] = 43
lut["motorcycle"] = 44
lut["mouse"] = 45
lut["orange"] = 46
lut["oven"] = 47
lut["parking meter"] = 48
lut["person"] = 49
lut["pizza"] = 50
lut["potted plant"] = 51
lut["refrigerator"] = 52
lut["remote"] = 53
lut["sandwich"] = 54
lut["scissors"] = 55
lut["sheep"] = 56
lut["sink"] = 57
lut["skateboard"] = 58
lut["skis"] = 59
lut["snowboard"] = 60
lut["spoon"] = 61
lut["sports ball"] = 62
lut["stop sign"] = 63
lut["suitcase"] = 64
lut["surfboard"] = 65
lut["teddy bear"] = 66
lut["tennis racket"] = 67
lut["tie"] = 68
lut["toaster"] = 69
lut["toilet"] = 70
lut["toothbrush"] = 71
lut["traffic light"] = 72
lut["train"] = 73
lut["truck"] = 74
lut["tv"] = 75
lut["umbrella"] = 76
lut["vase"] = 77
lut["wine glass"] = 78
lut["zebra"] = 79

def convert_coordinates(width, height, xmin, ymin, xmax, ymax):
    x_center = (xmin + xmax) / (2 * width)
    y_center = (ymin + ymax) / (2 * height)
    width = (xmax - xmin) / width
    height = (ymax - ymin) / height

    return (x_center, y_center, width, height)


def convert_xml2yolo( lut ):
    output_path = "/data/backup/pervinco_2020/darknet/build/darknet/x64/data/obj/COCO2017/labels"
    for fname in glob.glob("/data/backup/pervinco_2020/darknet/build/darknet/x64/data/obj/COCO2017/Annotations/*.xml"):
        filename = fname.split('/')[-1]
        filename = filename.split('.')[0]
        print(fname)

        tree = ET.parse(fname)
        root = tree.getroot()

        obj_xml = root.findall("object")
        size_xml = root.find('size')

        image_width = int(size_xml.find("width").text)
        image_hegiht = int(size_xml.find("height").text)

        if obj_xml[0].find('bndbox') != None:
            with open(output_path + '/' + filename + ".txt", 'w') as f:
                for obj in obj_xml:
                    class_id = obj.find("name").text
                    if class_id in lut:
                        label_str = str(lut[class_id])
                    else:
                        label_str = "-1"
                        print ("warning: label '%s' not in look-up table" %class_id)

                    bboxes = obj.find("bndbox")
                    xmin = float(float(bboxes.find('xmin').text))
                    ymin = float(float(bboxes.find('ymin').text))
                    xmax = float(float(bboxes.find('xmax').text))
                    ymax = float(float(bboxes.find('ymax').text))

                    # print(filename, image_width, image_hegiht, class_id, xmin, ymin, xmax, ymax)
                    result = convert_coordinates(image_width, image_hegiht, xmin, ymin, xmax, ymax)
                    # print(result)

                    f.write(label_str + " " + " ".join([("%.6f" % a) for a in result]) + '\n')

def main():
    convert_xml2yolo( lut )


if __name__ == '__main__':
    main()