# 🛰Environment
- Ubuntu - 18.04
- Docker - 19.03
- Tensorflow - 2.3 (2020.09 UPDATED!!!)
- Python - 3.6.9
- OpenCV-python - 4.1.1
         

# 🚀My Projects
## 🏗[Tensorflow](https://github.com/pervin0527/pervinco/blob/master/docs/Tensorflow_tutorial.md)  
   ### 1. tf.data input pipeline API image dataset 적용법
   - [Blog Post](https://www.notion.so/pervin0527/tf-data-API-27edd95e1f4c4c1eb352315125917d24)
   - [Jupyter-notebook Tutorial](https://gist.github.com/pervin0527/e9af4e0faab83243cb7f26990cac77f8)
   - [Source Code](https://github.com/pervin0527/pervinco/blob/master/source/tf2_image_classification.py)

   ### 2. tfrecord 생성
   - [Blog Post](https://www.notion.so/pervin0527/TFrecord-962fb914b74a47739b66e9e773e4817b)
   - [Source Code - feature를 csv파일에 작성하기](https://github.com/pervin0527/pervinco/blob/master/source/features_to_csv.py)
   - [Source Code - csv파일로 tfrecord파일 생성하기](https://github.com/pervin0527/pervinco/blob/master/source/generate_tfrecords.py)

## 🚁[Image Dataset Augmentation](https://github.com/pervin0527/pervinco/blob/master/docs/Image_Dataset_Augmentation.md)

   ### 1. Albumentations 소개

   <p align="left"><img src="./docs/doc_imgs/albumentation.jpeg" width=70%></p>

   - [Blog Post](https://www.notion.so/pervin0527/Albumentation-540221895eb04681969a43ee6d8acf71)

   ### 2. Image classification dataset에 Augmentation 적용하기

   <p align="left"><img src="./docs/doc_imgs/alb_graph4.png" width=70%></p>  

   - [Blog Post](https://www.notion.so/pervin0527/Augmentation-pipeline-for-image-classification-4932be16eb914e5892b015980efce4df)
   - [Source Code](https://github.com/pervin0527/pervinco/blob/master/source/albumentation_aug4.py)

   ### 3. Object Detecion dataset에 Augmentation 적용하기

   <table border="0">
   <tr>
      <td>
      <img src="./docs/doc_imgs/voc_aug1.png" width="200%" />
      </td>
      <td>
      <img src="./docs/doc_imgs/voc_aug2.png", width="200%" />
      </td>
   </tr>
   </table> 

   - [Blog Post](https://www.notion.so/pervin0527/Augmentation-pipline-for-Object-Detection-4e239d6db6eb4fe09da8b66f6af1ba4a)
   - [Source Code](https://github.com/pervin0527/pervinco/blob/master/source/albumentation_voc_aug2.py)  
   
         python3 albumentation_voc_aug.py \
         ./VOC2012/JPEGImages \     # Image dataset path
         ./VOC2012/Annotations \    # Annotation dataset path 
         ./VOC2012/Augmentations    # Path to save augmentation applied file

## ✈[Image Classification](https://github.com/pervin0527/pervinco/blob/master/docs/image_classification.md)   
   ### 1. Image 인식 개요 및 tensorflow 예제  
   - [About image recognition](http://research.sualab.com/introduction/2017/11/29/image-recognition-overview-1.html)
   - [Example inference](http://research.sualab.com/practice/2018/01/17/image-classification-deep-learning.html)
   - [Tensorflow 2.1 simple example](https://www.kaggle.com/philculliton/a-simple-tf-2-1-notebook)

   ### 2. Image Classification + EfficientNet

   <table border="0">
   <tr>
      <td>
      <img src="./docs/doc_imgs/img_0004.jpg" width="100%" />
      </td>
      <td>
      <img src="./docs/doc_imgs/img_0002.jpg", width="100%" />
      </td>
   </tr>
   </table>

   - [Blog Post](https://www.notion.so/pervin0527/Basic-Image-Classification-using-EfficientNet-8ac30bbd2bc84d4fb494740b5c7c99c6)
   - [Source Code - Training](https://github.com/pervin0527/pervinco/blob/master/source/Efnet_tf_data_train.py)
   - [Blog Post - Accuracy Test](https://github.com/pervin0527/pervinco/blob/master/source/tf2_model_test.py)

   ### 3. Multi Label Image Classification

   <table border="0">
   <tr>
      <td>
      <img src="./docs/doc_imgs/mlc.jpeg" width="100%" />
      </td>
      <td>
      <img src="./docs/doc_imgs/mlc2.png", width="100%" />
      </td>
   </tr>
   </table>

   - [Blog Post](https://www.notion.so/pervin0527/Multi-label-Classification-7a69efb0281c46cf80d2fe24e6a0f4b2)
   - [Source Code - training + tf.data](https://github.com/pervin0527/pervinco/blob/master/source/tf2_multi_label_classification.py)
   - [Source Code - training](https://github.com/pervin0527/pervinco/blob/master/source/multi_label_train.py) 
   - [Source Code - Accuracy Test](https://github.com/pervin0527/pervinco/blob/master/source/tf2_multi_label_predict.py)

## 🚝[Object Detection](https://github.com/pervin0527/pervinco/blob/master/docs/Object_Detection.md)  
   ### 1. Tensorflow Object Detection API 소개 및 사용방법
   
   <p align="center"><img src="./docs/doc_imgs/tensorflow2objectdetection.png"></p>

   - [Blog Post](https://www.notion.so/pervin0527/Tensorflow-2-Object-Detection-API-a354ee337107497dae8bcbde7341e2a8)
   - [Source Code - Test image inference](https://github.com/pervin0527/pervinco/blob/master/source/object_detection_image_inference.py)
   - [Source Code - Test Video inference](https://github.com/pervin0527/pervinco/blob/master/source/object_detection_video_inference.py)

   ### 2. Yolo V4

   <p align="center"><img src="./docs/doc_imgs/yolov4.png"></p>

   - ### [Blog Post](https://www.notion.so/pervin0527/YOLO-v4-d7d9a312e4b14005be22f393539b85cd)

   ### 3. Google/Automl - EfficientDet

   <p align="center"><img src="./docs/doc_imgs/efficientdet.png"></p>

   - [Blog Post](https://www.notion.so/pervin0527/EfficientDet-Google-AutoML-efc3927f229448759973322756c3bd23)

## InterMinds Projects
  ### 1. Smart checkout table - 2019.05 ~ 2019.12 Fin.  
   
   [Blog Post](https://www.notion.so/pervin0527/InterMinds-Smart-Checkout-Table-5c8bd2acc4b246eda8193a90bb8066f9)
   ![sco](./docs/doc_imgs/2.png)

  ### 2. Smart Shelf - 2020.01 ~ 2020.08 Fin.
   
   [Blog Post](https://www.notion.so/pervin0527/Interminds-Smart-Cabinet-c13f8aa64c144ebf8ead49506e0359d3)
   ![smart cabinet](./docs/doc_imgs/smart_cabinet_02.jpeg)
   