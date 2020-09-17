## 1. Tensorflow Object Detecion API

   <p align="center"><img src="/doc_imgs/tensorflow2objectdetection.png"></p>

   - ### Blog Post  
     1. [Install, Inference_test](https://pervin0527.github.io/tf2-object-detection/)
     2. [Training Custom dataset](https://pervin0527.github.io/tf2-object-detection-custom/)

   - ### Code
     - [Extract Features](https://github.com/pervin0527/pervinco/blob/master/source/create_training_files.py)
     - [Convert tfrecord](https://github.com/pervin0527/pervinco/blob/master/source/generate_tfrecords.py)
     - [Image Inference](https://github.com/pervin0527/pervinco/blob/master/source/object_detection_image_inference.py)
     - [Video Inference](https://github.com/pervin0527/pervinco/blob/master/source/object_detection_image_inference.py)

## 2. Yolo v4

   <p align="center"><img src="/doc_imgs/yolov4.png"></p>

   - ### [Official](https://github.com/AlexeyAB/darknet)
     - [Requirements](https://github.com/AlexeyAB/darknet#requirements)
     - [Compile using make](https://github.com/AlexeyAB/darknet#how-to-compile-on-linux-using-make)
     - Demo  
    
           ./darknet detector test cfg/coco.data cfg/yolov4.cfg yolov4.weights -thresh 0.25

     -  [How to Train](https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects)

  - ### [Blog post + Codes](https://pervin0527.github.io/YOLOv4/)


## 3. Google Automl/EfficientDet    
   - ### [Automl/EfficientDet GitHub](https://github.com/google/automl/tree/master/efficientdet)
    
   <p align="center"><img src="/doc_imgs/efficientdet.png"></p>  
   
   - ### Blog post  
     1. [Install, Pretrained_Inference, Train](https://pervin0527.github.io/efficientdet/)  
     2. [Train model inference](https://pervin0527.github.io/efficientdet2/)