import glob, os
import argparse

parser = argparse.ArgumentParser(description='Dataset split train/test')
parser.add_argument('--input_images_path', type=str)
parser.add_argument('--train_file_name', type=str, default='train.txt')
parser.add_argument('--test_file_name', type=str, default='test.txt')
args = parser.parse_args()

dataset_path = args.input_images_path
splited_path = dataset_path.split('/')[:-1]
output_path = '/'.join(splited_path)
print(output_path)

# Percentage of images to be used for the test set
percentage_test = 10

file_train = open(output_path + '/' + args.train_file_name, 'w')  
file_test = open(output_path + '/' + args.test_file_name, 'w')  

counter = 1  
index_test = round(100 / percentage_test)  
for pathAndFilename in glob.iglob(os.path.join(dataset_path, "*.jpg")):  
    title, ext = os.path.splitext(os.path.basename(pathAndFilename))

    if counter == index_test+1:
        counter = 1
        file_test.write(dataset_path + "/" + title + '.jpg' + "\n")
    else:
        file_train.write(dataset_path + "/" + title + '.jpg' + "\n")
        counter = counter + 1