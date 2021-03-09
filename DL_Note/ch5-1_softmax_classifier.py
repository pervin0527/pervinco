import tensorflow as tf
import numpy as np

def hypothesis(X):
    return tf.nn.softmax(tf.matmul(X, W) + b)

def cost_fn(X, Y):
    logits = hypothesis(X)
    cost = -tf.reduce_sum(Y * tf.math.log(logits), axis=1)
    cost_mean = tf.reduce_mean(cost)

    return cost_mean

def grad_fn(X, Y):
    with tf.GradientTape() as tape:
        cost = cost_fn(X, Y)
        grads = tape.gradient(cost, variables)

        return grads

def fit(X, Y, epochs=2000, verbose=100):
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

    for i in range(epochs):
        grads = grad_fn(X, Y)
        optimizer.apply_gradients(zip(grads, variables))

        if (i == 0) | ((i+1) % verbose == 0):
            print('Loss at epoch %d : %f' % (i+1, cost_fn(X, Y).numpy()))

if __name__ == "__main__":
    x_data = [[1, 2, 1, 1],
              [2, 1, 3, 2],
              [3, 1, 3, 4],
              [4, 1, 5, 5],
              [1, 7, 5, 5],
              [1, 2, 5, 6],
              [1, 6, 6, 6],
              [1, 7, 7, 7]]

    y_data = [[0, 0, 1],
              [0, 0, 1],
              [0, 0, 1],
              [0, 1, 0],
              [0, 1, 0],
              [0, 1, 0],
              [1, 0, 0],
              [1, 0, 0]]

    n_classes = 3
    x_data = np.asarray(x_data, dtype=np.float32)
    y_data = np.asarray(y_data, dtype=np.float32)
    print(x_data.shape)
    print(y_data.shape)

    W = tf.Variable(tf.random.normal([4, n_classes]), name='weight')
    b = tf.Variable(tf.random.normal([n_classes]), name='bias')
    variables = [W, b]
    print(variables)

    fit(x_data, y_data)

    b = hypothesis(x_data) # softmax를 거쳐 확률값 형태로 출력된 x_data
    print(b)
    print(tf.math.argmax(b, 1)) # One-hot encoding label에서 가장 큰 값의 index를 적어준 결과
    print(tf.math.argmax(y_data, 1))