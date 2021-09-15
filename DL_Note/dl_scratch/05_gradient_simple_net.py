import sys, os
import numpy as np

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 

    x = x - np.max(x) # 오버플로 대책
    return np.exp(x) / np.sum(np.exp(x))


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 값 복원
        it.iternext()   
        
    return grad


class SimpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss

network = SimpleNet()
# print(network.W)

x = np.array([0.6, 0.9])
p = network.predict(x)
# print(p)
print(np.argmax(p)) # 모델이 예측한 클래스 값

t = np.array([0, 0, 1]) # 정답 클래스는 2
# print(network.loss(x, t))

def f(W):
    return network.loss(x, t)

dW = numerical_gradient(f, network.W)
print(dW.shape)
print(dW)
"""
w11에 대한 L의 미분값이 0.11
이는 w11을 h만큼 늘리면 손실함수 값은 0.11h만큼 증가한다.

w23은 -0.34
w23을 h만큼 늘리면 손실함수 값은 -0.34h만큼 감소한다.

손실 함수를 줄인다는 관점에서 w23에 대한 L의 미분 값을 양의 방향으로 갱신하고,
w11은 음의 방향으로 갱신해야 함을 알 수 있다.
"""