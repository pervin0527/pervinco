import tensorflow as tf
tf.executing_eagerly()

"""
입력 값과 출력 값이 같은 값을 가지는 이유는 Weight, bias 값이 업데이트 되는 방향성을
사람이 먼저 계산해보고 올바른 방향으로 갈 수 있는지 보기 위함.
"""
x_data = [1, 2, 3, 4, 5]
y_data = [1, 2, 3, 4, 5]

"""
입력이 그대로 출력으로 나오기 위해서는 최종적으로 W = 1, b = 0에 가까운 값으로 나타나야 한다.
현재는 임의의 값을 넣어준 것이고, 원래는 random 값으로 초기화한다.
"""
W = tf.Variable(2.9)
b = tf.Variable(0.5)

hypothesis = W * x_data + b

"""
tf.square() : 제곱
tf.reduce_mean() : 결과 값의 차원이 줄어든 형태로 출력.
v = [1., 2., 3., 4.] # 1차원 값
tf.reduce_mean(v) # 2.5 결과적으로 0차원으로 줄어들었음.
"""

cost = tf.reduce_mean(tf.square(hypothesis - y_data))

"""
이제 필요한 것은 cost를 최소화하는 알고리즘의 선택이 필요하다.
"""
learning_rate = 0.01

"""
Gradient descent 알고리즘을 사용한다. - tf.GradientTape
변화되는 내용들을 tape에 기록하게 된다.
tape.gradient(cost, [W, b]) - cost 함수에 대한 변수들(W, b)의 경사도(미분값)을 구한다.
"""
for i in range(100+1):
    with tf.GradientTape() as tape:
        hypothesis = W * x_data + b
        cost = tf.reduce_mean(tf.square(hypothesis - y_data))

    W_grad, b_grad = tape.gradient(cost, [W, b])


    """
    A.assign_sub(B) == A -= B 를 통해서
    가중치 업데이트
    learning rate 값을 곱해서 weight, bias를 각각 업데이트 해준다.
    즉, 앞서 구했던 기울기 값을 얼마만큼 반영할 것인가를 결정
    """
    W.assign_sub(learning_rate * W_grad)
    b.assign_sub(learning_rate * b_grad)

    if i % 10 == 0:
        print("{:5}|{:10.4f}|{:10.4}|{:10.6f}".format(i, W.numpy(), b.numpy(), cost))

""" Predict """
print(W * 5 + b)
print(W * 2.5 + b)