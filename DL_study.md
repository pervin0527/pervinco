Deep Learning Study
===================

### Check out trends in deep learning 
 - <http://www.guide2research.com/topconf/>

### Mathematics
딥러닝을 하면서 필요한 수학적인 개념들을 정리해 보았다.

첫번째로 평균(mean), 분산(variance), 편차(deviation), 표준편차(standard deviation) 이다.
![math](doc_imgs/math.png "mathematics")

참조 - [LearnX 통계 블로그](https://learnx.tistory.com/entry/%ED%86%B5%EA%B3%84%EC%9D%98-%EA%B8%B0%EC%B4%88%EC%9D%B8-%ED%8F%89%EA%B7%A0-%EB%B6%84%EC%82%B0-%ED%91%9C%EC%A4%80%ED%8E%B8%EC%B0%A8)

### Data Normalize
내가 주로 이용하는 image data는 pixel map을 구성하는 각각의 pixel은 0 ~ 255 사이 값을 가진다. 
이를 255로 나누어 주면 0 ~ 1사이 값들로 구성되게 된다.

![Normalize](doc_imgs/normalize.jpeg "normalize")

위 이미지에서 보이는 것처럼 Unnormalized data는 학습하는 과정이 매우 복잡하고 길다.
단순하게 생각해도 값이 0 ~ 255 사이 값이니까.

Normalize를 적용해주면 0 ~ 1사이 값이니 학습이 간결해지고, 추가적으로 **local optimum** 에빠지게 될 가능성도 줄이는 효과가 있다.

![local_minimum](doc_imgs/local_minimum.png "local_minum")
