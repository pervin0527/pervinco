import numpy as np

class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0) # 입력인 x의 원소 값이 0 이하인 인덱스는 True 그 외(0보다 큰 원소)는 False
        print(self.mask)

        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0 # 순전파 때 만들어둔 mask를 써서 원소가 True인 곳에는 상류에서 전파된 dout을 0으로 처리.
        dx = dout

        return dx

x = np.array([[1.0, -0.5], [-2.0, 3.0]])
test = Relu()
test_forward = test.forward(x)
test_back_ward = test.backward(test_forward)
print(test_forward)
print(test_back_ward)

class Sigmoid:
    def __init__(self):
        sefl.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out

        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx