import numpy as np
import tensorflow as tf

# data and label(정답)
data = np.array([
    [73., 80., 75., 152.],
    [93., 88., 93., 185.],
    [89., 91., 90., 180.],
    [96., 98., 100., 196.],
    [73., 66., 70., 142.],
], dtype=np.float32)

# 아무것도 입력하지 않고 :로 사용할 경우 모든 행 또는 열을 뜻한다.
X = data[:, :-1] # 모든 행, 열 중에서 마지막 열을 제외하고 가져오겠다.
y = data[:, [-1]] # 모든 행, 마지막 열만 가져오겠다.
print(X)
print(y)

W = tf.Variable(tf.random.normal([3, 1])) # 행렬 곱 연산을 하므로 행의 값을 X의 열과 맞추고, 열은 출력 y처럼 1개의 값을 출력하기 때문에 1
print(W)
b = tf.Variable(tf.random.normal([1]))
print(b)

learning_rate = 0.000001

n_epochs = 2000
for i in range(n_epochs + 1):
    with tf.GradientTape() as tape:
        hypothesis = tf.matmul(X, W) + b
        cost = tf.reduce_mean((tf.square(hypothesis - y)))

        W_grad, b_grad = tape.gradient(cost, [W, b])
        W.assign_sub(learning_rate * W_grad)
        b.assign_sub(learning_rate * b_grad)

    if i % 100 == 0:
        print("{:5} | {:10.4f}".format(i, cost.numpy()))