"""
x_data가 3차원 배열이기에 3차원 공간에 표현하여 x1과 x2, x3를 학습하고, y_data 3개 클래스를 구분하는 예제입니다
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.random.set_seed(777)
# 해당 Data를 3개의 색상으로 구분해서 출력해 보겠습니다.(파랑 = 2, 초록 = 1, 빨강 = 0)
x_train = [[1, 2, 1],
           [1, 3, 2],
           [1, 3, 4],
           [1, 5, 5],
           [1, 7, 5],
           [1, 2, 5],
           [1, 6, 6],
           [1, 7, 7]]

# One_hot encoding 된 label
# [0,0,1] = 파랑, [0,1,0] = 초록, [1,0,0] = 빨강
y_train = [[0, 0, 1],
           [0, 0, 1],
           [0, 0, 1],
           [0, 1, 0],
           [0, 1, 0],
           [0, 1, 0],
           [1, 0, 0],
           [1, 0, 0]]

x_test = [[2, 1, 1],
          [3, 1, 2],
          [3, 3, 4]]

y_test = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1]]

x1 = [x[0] for x in x_train]
x2 = [x[1] for x in x_train]
x3 = [x[2] for x in x_train]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1, x2, x3, c=y_train, marker='^')

ax.scatter(x_test[0][0], x_test[0][1], x_test[0][2], c="black", marker='^')
ax.scatter(x_test[1][0], x_test[1][1], x_test[1][2], c="black", marker='^')
ax.scatter(x_test[2][0], x_test[2][1], x_test[2][2], c="black", marker='^')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()

dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(len(x_train))
W = tf.Variable(tf.random.normal((3, 3)))
b = tf.Variable(tf.random.normal((3,)))


def softmax_fn(features):
    hypothesis = tf.nn.softmax(tf.matmul(features, W) + b)

    return hypothesis

def loss_fn(hypothesis, labels):
    cost = tf.reduce_mean(-tf.reduce_sum(labels * tf.math.log(hypothesis), axis=1))

    return cost

is_decay = True
starter_learning_rate = 0.1
    
if is_decay:    
    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=starter_learning_rate,
                                                                   decay_steps=1000,
                                                                   decay_rate=0.96,
                                                                   staircase=True)
    optimizer = tf.keras.optimizers.SGD(learning_rate)

else:
    optimizer = tf.keras.optimizers.SGD(learning_rate=starter_learning_rate)

def grad(hypothesis, features, labels):
    with tf.GradientTape() as tape:
        hypothesis = softmax_fn(features)
        loss_value = loss_fn(hypothesis, labels)

    return tape.gradient(loss_value, [W,b])

def accuracy_fn(hypothesis, labels):
    prediction = tf.argmax(hypothesis, 1)
    is_correct = tf.equal(prediction, tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

    return accuracy


EPOCHS = 1001

for step in range(EPOCHS):
    for features, labels  in iter(dataset):
        features = tf.cast(features, tf.float32)
        labels = tf.cast(labels, tf.float32)

        grads = grad(softmax_fn(features), features, labels)
        optimizer.apply_gradients(grads_and_vars=zip(grads,[W,b]))

        if step % 100 == 0:
            print("Iter: {}, Loss: {:.4f}".format(step, loss_fn(softmax_fn(features), labels)))

x_test = tf.cast(x_test, tf.float32)
y_test = tf.cast(y_test, tf.float32)
test_acc = accuracy_fn(softmax_fn(x_test), y_test)

print("Testset Accuracy: {:.4f}".format(test_acc))