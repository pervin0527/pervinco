## 1. ImageClassification using Tensorflow

   <table border="0">
   <tr>
      <td>
      <img src="./doc_imgs/img_0004.jpg" width="100%" />
      </td>
      <td>
      <img src="./doc_imgs/img_0002.jpg", width="100%" />
      </td>
   </tr>
   </table>

   - [Blog](https://www.notion.so/pervin0527/Basic-Image-Classification-using-EfficientNet-8ac30bbd2bc84d4fb494740b5c7c99c6)

   -  Concept
      - [About image recognition](http://research.sualab.com/introduction/2017/11/29/image-recognition-overview-1.html)
      - [Example inference](http://research.sualab.com/practice/2018/01/17/image-classification-deep-learning.html)

   - Tensorflow
      - [Tensorflow tutorial](https://github.com/pervin0527/pervinco/blob/master/tensorflow_tutorial.md)
      - [Tensorflow 2.1 simple example](https://www.kaggle.com/philculliton/a-simple-tf-2-1-notebook)

   - Source Code (2020.09.17 update)
     - Tensorflow 2.3에서 EfficientNet API 내장
     - [EfficientNet with tf.data](https://github.com/pervin0527/pervinco/blob/master/source/Efnet_tf_data_train.py)
     - [Measuring model performance](https://github.com/pervin0527/pervinco/blob/master/source/tf2_model_test.py)

   - [2020.06.11 UPDATE - Learning Rate Schedule Callback function Added](https://github.com/pervin0527/pervinco/blob/05ba90f7a1921ddc84c79f3be8c232119de0b0e6/source/Efnet_tf_data_train.py#L147)
   - [2020.09.17 UPDATE - EfficientNet Train with Multi_GPU](https://github.com/pervin0527/pervinco/blob/master/source/Efnet_multi_gpu_train.py)


## 2. Multi Label ImageClassification

   <table border="0">
   <tr>
      <td>
      <img src="./doc_imgs/mlc.jpeg" width="100%" />
      </td>
      <td>
      <img src="./doc_imgs/mlc2.png", width="100%" />
      </td>
   </tr>
   </table>

   - [Blog post](https://www.notion.so/pervin0527/Multi-label-Classification-7a69efb0281c46cf80d2fe24e6a0f4b2)
   - [Reference](https://www.analyticsvidhya.com/blog/2019/04/build-first-multi-label-image-classification-model-python/)
   - Source Code  
     - [Training](https://github.com/pervin0527/pervinco/blob/master/source/multi_label_train.py)  
     - [Predict](https://github.com/pervin0527/pervinco/blob/master/source/tf2_multi_label_predict.py)  
     - [Using tf.data training](https://github.com/pervin0527/pervinco/blob/master/source/tf2_multi_label_classification.py)
