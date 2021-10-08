import glob, os
import argparse

parser = argparse.ArgumentParser(description='Dataset split train/test')
parser.add_argument('--input_images_path', type=str)
parser.add_argument('--file_name', type=str, default='train.txt')
parser.add_argument('--output_path', type=str)
args = parser.parse_args()

dataset_path = args.input_images_path
splited_path = dataset_path.split('/')[:-1]
output_path = args.output_path
print(output_path)

file_train = open(output_path + '/' + args.file_name, 'w')  

for pathAndFilename in glob.iglob(os.path.join(dataset_path, "*.jpg")):  
    title, ext = os.path.splitext(os.path.basename(pathAndFilename))
    file_train.write(dataset_path + "/" + title + '.jpg' + "\n")