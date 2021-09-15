import numpy as np

# def AND(x1, x2):
#     x = np.array([x1, x2])
#     w = np.array([0.5, 0.5])
#     b = -0.7

#     tmp = np.sum(w * x) + b
#     print(tmp)

#     if tmp <= 0:
#         return 0

#     else:
#         return 1

# x = np.array([0, 1])
# w = np.array([0.5, 0.5])
# b = -0.7

# print(np.sum(w * x) + b)
# print(AND(0, 1))

def sigmoid(a):
    return 1 / (1 + np.exp(-a))

# t = np.array([-1.0, 1.0, 2.0, 3.0])
# print(sigmoid(t))

def ReLU(x):
    return np.maximum(0, x)

# print(ReLU(t))
# print(ReLU(np.array([-2, -1, 0, 1])))

# print(np.ndim(np.array([-2, -1, 0, 1]))) # 배열의 차원 수 출력 np.ndim()

# A = np.array([[1, 2, 3], [4, 5, 6]])
# B = np.array([[1, 2], [3, 4], [5, 6]])
# print(A.shape, B.shape)
# print(np.dot(A,B)) # 행렬의 곱셈 np.dot()

def identity_function(x):
    return x

X = np.array([1.0, 0.5])

W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])
A1 = np.dot(X, W1) + B1
Z1 = sigmoid(A1)

W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])
A2 = np.dot(Z1, W2) + B2
Z2 = sigmoid(A2)

W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0.1, 0.2])
A3 = np.dot(Z2, W3) + B3
Y = identity_function(A3)