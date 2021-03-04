import tensorflow as tf

tf.random.set_seed(0) # random seed 초기화. 다음에 이 코드를 수행했을 때도 이전 때와 동일한 결과를 얻기 위함.
x_data = [1., 2., 3., 4.]
y_data = [1., 3., 5., 7.]

W = tf.Variable(tf.random.normal([1], -100., 100.)) # 정규분포를 따르는 랜덤한 값을 [1] shape으로 받겠다.
print(W)

for step in range(300):
    hypothesis = W * x_data
    cost = tf.reduce_mean(tf.square(hypothesis - y_data))

    learning_rate = 0.01
    gradient = tf.reduce_mean(tf.multiply(tf.multiply(W, x_data) - y_data, x_data)) # sum((hypothesis - y_data) * x_data) / len(x_data)
    descent = W - tf.multiply(learning_rate, gradient) # W - (alpha * gradient)
    W.assign(descent)

    if step % 10 == 0:
        print('{:5} | {:10.4f} | {:10.6f}'.format(step, cost.numpy(), W.numpy()[0]))