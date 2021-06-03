import os
import cv2
import pathlib
import pandas as pd
import argparse
from tqdm import tqdm

def get_annotations_dataframe(annotations_bbox, class_id, file_name):
    try:
        filter = annotations_bbox.ImageID.eq(file_name) & annotations_bbox.LabelName.eq(class_id)
        return annotations_bbox.loc[filter]

    except:
        raise Exception('error get annotations dataframe!')

def get_class_id(class_descriptions_boxable, class_name):
    class_filter = class_descriptions_boxable['Tortoise'].isin([class_name])
    filtered_dataframe = class_descriptions_boxable.loc[class_filter]

    try:
        return filtered_dataframe.to_numpy()[0][0]

    except:
        raise Exception('{} not exists!'.format(class_name))


def create_darknet_annotation(df, file_name, class_name_index, output_dir):
#   print('\n===== {}'.format(file_name))
  with open(f'{output_dir}/{file_name}.txt', 'w') as txt_file:
    for arr_values in df[['XMin', 'XMax', 'YMin', 'YMax']].to_numpy():
        x_values = [float(arr_values[0]), float(arr_values[1])]
        y_values = [float(arr_values[2]), float(arr_values[3])]

        center_x = (x_values[1] + x_values[0]) / 2
        center_y = (y_values[1] + y_values[0]) / 2

        w = (x_values[1] - x_values[0])
        h = (y_values[1] - y_values[0])

        # print("{} {} {} {} {}".format(class_name_index, center_x, center_y, w, h))
        txt_file.write("{} {} {} {} {}\n".format(class_name_index, center_x, center_y, w, h))


def make_ds():
    if not os.path.isdir(f'{output_path}'):
        os.makedirs(f'{output_path}')

    for (class_name_index, class_name) in enumerate(sorted(labels)):

        if class_name == 'Man' or class_name == 'Woman':
            class_name_index = labels.index('Person')

        elif class_name == 'Table':
            class_name_index = labels.index('Desk')

        elif class_name == 'Television':
            class_name_index = labels.index('Computer monitor')

        class_id = get_class_id(class_descriptions_boxable, class_name)
        # print(class_name_index, class_name, class_id)
        
        images = list(ds_path.glob(f'{class_name}/*.jpg'))
        images = [str(path) for path in images]

        print(class_name_index, class_name)
        for i in tqdm(range(len(images))):
            img_file = images[i]
            
            file_name = (img_file.split('/')[-1]).split('.')[0]
            file_dir = img_file.split('/')[:-1] 
            file_dir = '/'.join(file_dir)

            save_name = f'{class_name}_{i}'
            image = cv2.imread(img_file)
            cv2.imwrite(f'{output_path}/{save_name}.jpg', image)
            
            df = get_annotations_dataframe(train_annotations_bbox, class_id, file_name)
            create_darknet_annotation(df, save_name, class_name_index, output_path)

            if i == 4000:
                break

        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='OID dataset post-processing')
    parser.add_argument('--oid_path', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--is_train', type=str, default=True)
    args = parser.parse_args()

    dataset_path = args.oid_path
    output_path = args.output_path

    train_annotations_bbox = pd.read_csv(f'{dataset_path}/csv_folder/train-annotations-bbox.csv')
    class_descriptions_boxable = pd.read_csv(f'{dataset_path}/csv_folder/class-descriptions-boxable.csv')

    is_train = ''
    if args.is_train:
        is_train = 'train'

    else:
        is_train = 'valid'

    ds_path = pathlib.Path(f'{dataset_path}/Dataset/{is_train}')
    labels = sorted(item.name for item in ds_path.glob('*/') if item.is_dir())
    print(labels)

    make_ds()