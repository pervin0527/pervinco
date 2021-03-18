import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# 대부분의 값들이 800대 값인데, 일부 엄청나게 큰 값들이 존재.
# 그래프로 나타내보면, 800대 값들이 0에 분포한 것으로 표시가 된다.
# 따라서 Normalization이 필요
xy = np.array([[828.659973, 833.450012, 908100, 828.349976, 831.659973],
               [823.02002, 828.070007, 1828100, 821.655029, 828.070007],
               [819.929993, 824.400024, 1438100, 818.97998, 824.159973],
               [816, 820.958984, 1008100, 815.48999, 819.23999],
               [819.359985, 823, 1188100, 818.469971, 818.97998],
               [819, 823, 1198100, 816, 820.450012],
               [811.700012, 815.25, 1098100, 809.780029, 813.669983],
               [809.51001, 816.659973, 1398100, 804.539978, 809.559998]])

x_train = xy[:, 0:-1]
y_train = xy[:, [-1]]

plt.plot(x_train, 'ro')
plt.plot(y_train)
plt.show()

def normalization(data):
    numerator = data - np.min(data, 0)
    denomiator = np.max(data, 0) - np.min(data, 0)

    return numerator / denomiator

xy = normalization(xy)
print(xy)
x_train = xy[:, 0:-1]
y_train = xy[:, [-1]]

plt.plot(x_train, 'ro')
plt.plot(y_train)

plt.show()

dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(len(x_train))
W = tf.Variable(tf.random.normal((4, 1)), dtype=tf.float32)
b = tf.Variable(tf.random.normal((1,)), dtype=tf.float32)

def linearReg_fn(features):
    hypothesis = tf.matmul(features, W) + b

    return hypothesis

def l2_loss(loss, beta = 0.01):
    W_reg = tf.nn.l2_loss(W) # output = sum(t ** 2) / 2
    loss = tf.reduce_mean(loss + W_reg * beta)

    return loss

def loss_fn(hypothesis, labels, flag = False):
    cost = tf.reduce_mean(tf.square(hypothesis - labels))
    if flag:
        cost = l2_loss(cost)

    return cost

is_decay = True
starter_learning_rate = 0.1

if is_decay:
    # epoch이 50번째 마다 기존 learning rate * 0.96
    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=starter_learning_rate,
                                                                   decay_steps=50,
                                                                   decay_rate=0.96,
                                                                   staircase=True)
    optimizer = tf.keras.optimizers.SGD(learning_rate)

else:
    optimizer = tf.keras.optimizers.SGD(learning_rate=starter_learning_rate)

def grad(hypothesis, labels, l2_flag):
    with tf.GradientTape() as tape:
        hypothesis = linearReg_fn(features)
        loss_value = loss_fn(hypothesis, labels, l2_flag)

    return tape.gradient(loss_value, [W,b]), loss_value


EPOCHS = 101

for step in range(EPOCHS):
    for features, labels  in dataset:
        features = tf.cast(features, tf.float32)
        labels = tf.cast(labels, tf.float32)
        grads, loss_value = grad(linearReg_fn(features), labels, True)
        optimizer.apply_gradients(grads_and_vars=zip(grads,[W,b]))

    if step % 10 == 0:
        print("Iter: {}, Loss: {:.4f}".format(step, loss_value))