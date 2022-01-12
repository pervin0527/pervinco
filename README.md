# Pervinco's Repo
Vision Deep Learning을 해보면서 만든 내용들을 담아둔 저장소입니다.

## Requirements
- Tensorflow >= 2.5 
- Albuemntations
- OpenCV
- Python3
- Pandas
- Numpy

## Image Data Augmentation
1. Cvat을 이용해서 annotation한 데이터를 PASCAL VOC로 변환하기 - [cvat processing](source/augmentation/cvat_data.py)
2. image와 annotation file Augmentation하기(CutMix & MixUp & Mosaic) - [detection data augmentation](source/augmentation/detection_data_augmentation.py)
3. PASCAL VOC 데이터를 yolo 형식(또는 반대로)으로 변환하기  
   - [voc2yolo](source/augmentation/xml2txt.py)
   - [yolo2voc](source/augmentation/txt2xml.py)

## Image Classification
1. tf.keras API에서 사용가능한 모델을 이용. - [image_classification with EfficientNet](source/image_classification/EfficientNet_ver1.py)  
2. CutMix & MixUp을 적용하여 학습 - [apply CutMix & MixUp](source/image_classification/cut_mix_training.py)  
3. Model SubClassing API 사용
      - [subclassing 기초](source/image_classification/model_subclassing_basic.py)
      - [subclassing + model.fit()](source/image_classification/model_subclass_fit.py)
      - [subclassing + gradient.tape()](source/image_classification/model_subclass_tape.py)

## Object Detection
1. Tensorflow Object Detection API - [README](source/object_detection/README.md)  
  
2. Tensorflow lite Model Maker - [Model Maker](source/object_detection/tensorflow_object_detection/model_maker_train.py)