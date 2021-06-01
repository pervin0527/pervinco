import os
import cv2
import pathlib

def mkdir_mv_files(classes):
    for c in classes:
        images = list(ds_path.glob(f'{c}/*.jpg'))
        images = sorted([str(path) for path in images])
        total_images = len(images)

        coords = ds_path.glob(f'{c}/Label/*.txt')
        coords = sorted([str(path) for path in coords])
        total_coords = len(coords)

        print(c, total_images, total_coords)
        
        idx = 0
        for image, coord in zip(images, coords):
            image_name = image.split('/')[-1]
            image_name = image_name.split('.')[0]

            coord_name = coord.split('/')[-1]
            coord_name = coord_name.split('.')[0]

            if image_name != coord_name:
                print('Mapping ERROR')
                print(image, coord)
                exit()

            save_path = f'{output_path}/{dataset_name}'
            if not os.path.isdir(save_path):
                os.makedirs(save_path)

            image = cv2.imread(image)
            cv2.imwrite(f'{save_path}/{c}_{idx}.jpg', image)

            coord = open(coord, 'r')
            lines = coord.readlines()
            
            f = open(f'{save_path}/{c}_{idx}.txt', 'w')
            for line in lines:
                f.write(line)

            f.close()
            idx += 1
            

if __name__ == "__main__":
    dataset_name = 'in_office'
    output_path = '/data/datasets/OIDv4_ToolKit/sorted'
    dataset_path = '/data/datasets/OIDv4_ToolKit/OID/Dataset/train'
    ds_path = pathlib.Path(dataset_path)

    labels = sorted(item.name for item in ds_path.glob('*/') if item.is_dir())
    mkdir_mv_files(labels)