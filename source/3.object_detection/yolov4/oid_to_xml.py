import os
import cv2
import glob
import pathlib
import argparse
import time
import datetime
from tqdm import tqdm
from lxml import etree

def write_xml():
    DIRS = sorted(glob.glob(f'{dataset_path}/*'))
    
    for INDEX, DIR in enumerate(DIRS):
        if " " in DIR:
            os.rename(DIR, DIR.replace(" ", "_"))

        print("\n" + "Creating PASCAL VOC XML Files for Class:", DIR)

        f_path = pathlib.Path(DIR)

        images = list(f_path.glob('*.jpg'))
        images = [str(path) for path in images]

        # labels = list(f_path.glob('Label/*.txt'))
        # labels = [str(path) for path in labels]

        label = DIR.split('/')[-1]

        if label == 'Computer_monitor':
            label = 'Television'

        elif label == "Person" or label == 'Woman':
            label = 'Man'

        elif label == 'Desk':
            label = 'Table'

        for idx in tqdm(range(len(images))):
            # sub = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

            filename = images[idx]
            filename_str = (filename.split('/')[-1]).split('.')[0]

            image = cv2.imread(filename)
            cv2.imwrite(f'{output_path}/{label}_{INDEX}_{idx}.jpg', image)       

            annotation = etree.Element("annotation")

            folder = etree.Element('folder')
            folder.text = filename.split('/')[-2]
            annotation.append(folder)

            filename_xml = etree.Element("filename")
            filename_xml.text = f'{label}_{INDEX}_{idx}.jpg'
            annotation.append(filename_xml)

            path = etree.Element("path")
            path.text = '/'.join(filename.split('/')[:-1]) + f'/{filename_xml.text}'
            annotation.append(path)

            source = etree.Element("source")
            annotation.append(source)

            database = etree.Element("database")
            database.text = "Unknown"
            source.append(database)

            size = etree.Element("size")
            annotation.append(size)

            width = etree.Element("width")
            height = etree.Element("height")
            depth = etree.Element("depth")

            width.text = str(image.shape[1])
            height.text = str(image.shape[0])
            depth.text = str(image.shape[2])

            size.append(width)
            size.append(height)
            size.append(depth)

            label_original = open(f'{"/".join(filename.split("/")[:-1])}/Label/{filename_str}.txt', 'r')

            for line in label_original:
                line = line.strip()
                l = line.split(' ')
                class_name = l[0]
                try:
                    xmin_l = str(int(float(l[1])))
                    add1 = 0
                except ValueError:
                    class_name = l[0]+"_"+l[1]
                    add1 = 1

                xmin_l = str(int(float(l[1+add1])))
                ymin_l = str(int(float(l[2+add1])))
                xmax_l = str(int(float(l[3+add1])))
                ymax_l = str(int(float(l[4+add1])))
                
                obj = etree.Element("object")
                annotation.append(obj)

                name = etree.Element("name")
                # name.text = class_name
                name.text = label
                obj.append(name)

                pose = etree.Element("pose")
                pose.text = "Unspecified"
                obj.append(pose)

                truncated = etree.Element("truncated")
                truncated.text = "0"
                obj.append(truncated)

                difficult = etree.Element("difficult")
                difficult.text = "0"
                obj.append(difficult)

                bndbox = etree.Element("bndbox")
                obj.append(bndbox)

                xmin = etree.Element("xmin")
                xmin.text = xmin_l
                bndbox.append(xmin)

                ymin = etree.Element("ymin")
                ymin.text = ymin_l
                bndbox.append(ymin)

                xmax = etree.Element("xmax")
                xmax.text = xmax_l
                bndbox.append(xmax)

                ymax = etree.Element("ymax")
                ymax.text = ymax_l
                bndbox.append(ymax)

            s = etree.tostring(annotation, pretty_print=True)
            with open(f'{output_path}/{label}_{INDEX}_{idx}.xml', 'wb') as f:
                f.write(s)
                f.close()

            # break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OID xml maker")
    parser.add_argument('--input_ds', type=str)
    parser.add_argument('--output_path', type=str)
    args = parser.parse_args()

    dataset_path = args.input_ds
    output_path = args.output_path
    print(dataset_path)

    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    write_xml()