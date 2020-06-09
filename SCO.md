---
title: "Interminds Smart Checkout Table"
excerpt: "스마트 계산대"
toc: true
toc_sticky: true
 
last_modified_at: 2020-06-08
---
이번 포스트에서는 내가 회사에서 진행했던 프로젝트에 대해 소개하고자 한다.

2019년 초에서 2019년 말까지 진행된 프로젝트로, 초기 버전에서 문제가 많은 것을 지속적인 개선을 통해 좋은 성능을 나타내게끔 만들었다.

Smart Checkout Table이라는 이름으로, 현재 대형마트에서 적용한 셀프 계산대의 상위호환으로 진행한 프로젝트이다.  

# System Architecture
먼저 전체 시스템 구조도에 대해 그림을 통해 살펴보자.  
![](./doc_img/Untitled%20Diagram.jpg)

1. 사용자가 구매하려는 제품을 테이블 위에 펼쳐놓는다.
2. Web cam을 위에서 아래로 테이블을 내려다보는 형태로 설치하고, 지속적으로 Frame을 받도록한다.
3. 1장의 frame을 읽고 나면 해당 frame내에서 Object Detection을 통해 찾을 수 있는 모든 Object의 loacation을 탐지한다.
4. 얻은 bbox들을 통해 Image Classification model에 넣고 각 box에 대한 예측 값을 얻어낸다.
5. 구매하려는 전체 제품들에 대한 예측 값을 UI를 통해 사용자에게 출력

# Details

- ### Object Detection
    앞서 System Architecture에서 본 내용들 중 2번까지는 쉽게 이해할 수 있을 것이라 생각한다.  
    Object Detction을 위해 적용한 모델은 [**Yolo V3**](https://pjreddie.com/darknet/yolo/)이다. Vision Deep Learning을 하는 사람이라면 누구나 한번쯤 접해보았을 모델이다. 성능도 당시 준수했으며, 누구나 쉽게 이용 가능한 장점이 있다고 생각한다.  

    하지만 아쉽게도, Yolo V3를 비롯한 대부분의 Object Detection model은 Train, Validation dataset을 구축할 때, Annotation 작업으로 인해 상품의 변동사항에 유연하게 대처할 수 없다는 단점이 있다.
    ![](./doc_img/demo3.jpg)
        
    **내가 적용하는 분야는 Retail부분으로**   
        1. <U>신제품이 자주 출시되어</U> 새로운 image 촬영과 label추가 annotation작업을 해야하는 경우  
        2. 같은 제품이라 하더라도 판매 <U>시기에 따라 제품의 디자인이 수시로 변경되기 때문에</U> label에 변경이 없어도 Image를 다시 촬영하여 학습시켜야 하는 경우  
        3. 마트의 경우 지정된 공휴일 외에는 계속해서 오픈하고 고객들에게 서비스를 해야하기 때문에 model이 update되어야할 시간적 여유가 짧음.  
        4. 제품 수가 100가지 이하이면 시간과 비용을 투자해 작업을 진행할 수 있다고 해도,
    1000가지, 2000가지가 넘는 상품들을 보유한 대형마트의 경우, 소모될 시간과 비용을 생각하면....

    이러한 이유들로 인해, Object Detection 모델 내에서 모든 label들을 적용해서 학습시키는 것은 서비스 측면에서 비효율적이라 판단하여, **Product라는 1개의 label로 모든 제품들을 labeling하였다.**

    Detection에서 "Product" 단일 label로 학습시킨 모델을 inference test한 이미지이다.  
    ![](./doc_img/detection_infer.png)

- ### BBox Classification
    상대적으로 Image labeling 및 Preprocessing에 투자되는 시간이 짧으면서 제품이 추가되거나 변동되어도 Update가 적용하기 용이한 Image Classification model에서는 각각 제품마다의 label을 지정해 학습시켰다.

    1. Product 단일 label로 여러가지 제품들을 학습시킨 Detection model이 있으면, classification을 위한 dataset을 구축하는데 있어서 Auto Crop이 가능하기 때문에 용이하다.
    2. Crop한 데이터들을 각각 barcode 폴더에 맞게 넣어주고, Augmentation을 적용해준다.
    3. Smart Checkout Table의 경우 제품이 어떤 면으로, 어떤 위치에 놓일지 모르기 때문에 아래 옵션들은 기본적으로 적용해주도록 한다.
        - Resize(224, 224) 
        - Rotate(360도 limit)
        - Width_Shift, Height_Shift  

    ![](./doc_img/images/products.png)

    사용한 Model은 ResNet50으로 Tensorflow Keras 프레임워크에서 사용하였다.  
    소스코드는 아래 링크를 통해 볼 수 있다.  
    [https://github.com/pervin0527/pervinco/blob/master/source/keras_image_classification.py](https://github.com/pervin0527/pervinco/blob/master/source/keras_image_classification.py)

    Detection Inference에서 얻은 Bounding Box의 수는 5가지. 출력된 label명은 전부 product임을 볼 수 있다.  
    그 다음 Classification model에 각 Bounding Box들을 input으로 사용해 Classification을 진행해 해당 상품이 무엇인지 분류하는 작업을 한다.  
    ![](./doc_img/classification_infer.png)

    
 - ### User Interface
    이후 사용자에게 UI를 통해 결과물을 보여주도록 한다.
    ![](./doc_img/2.png)

전체 Source Code는 다음 링크에서 확인할 수 있다.
[https://gist.github.com/pervin0527/e65de5a3d73a4a4b76b5b805ea761399](https://gist.github.com/pervin0527/e65de5a3d73a4a4b76b5b805ea761399)