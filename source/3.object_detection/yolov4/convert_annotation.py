import pandas as pd
import os
from glob import glob

def get_annotations_dataframe(annotations_bbox, class_id, file_name):
  try:
    # the same file can be used for another type of class,
    # this filter find the file of our class
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
  print('\n===== {}'.format(file_name))
  with open(os.path.join(output_dir, file_name + '.txt'), 'w') as txt_file:
    for arr_values in df[['XMin', 'XMax', 'YMin', 'YMax']].to_numpy():
      x_values = [float(arr_values[0]), float(arr_values[1])]
      y_values = [float(arr_values[2]), float(arr_values[3])]

      center_x = (x_values[1] + x_values[0]) / 2
      center_y = (y_values[1] + y_values[0]) / 2

      w = (x_values[1] - x_values[0])
      h = (y_values[1] - y_values[0])

      print("{} {} {} {} {}".format(class_name_index, center_x, center_y, w, h))
      txt_file.write("{} {} {} {} {}\n".format(class_name_index, center_x, center_y, w, h))


# load annotations bounding boxes
train_annotations_bbox = pd.read_csv('/data/datasets/OIDv4_ToolKit/OID/csv_folder/train-annotations-bbox.csv')

# load classes csv file
class_descriptions_boxable = pd.read_csv('/data/datasets/OIDv4_ToolKit/OID/csv_folder/class-descriptions-boxable.csv')

for (class_name_index, class_name) in enumerate(sorted(os.listdir('/data/datasets/OIDv4_ToolKit/OID/Dataset/train'))):
  class_id = get_class_id(class_descriptions_boxable, class_name)
  for img_file in glob('/data/datasets/OIDv4_ToolKit/OID/Dataset/train/{}/*.jpg'.format(class_name)):
    file_name = os.path.basename(img_file).split('.')[0]
    file_dir = os.path.dirname(img_file)
    df = get_annotations_dataframe(train_annotations_bbox, class_id, file_name)
    create_darknet_annotation(df, file_name, class_name_index, file_dir)