import numpy as np

def cross_entropy(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta)) # log 함수에 0을 곱하게 되면 무한대로 발산되기 때문에 delta라는 아주 작은 값을 더해서 0이 되는 것을 방지한다.

t = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
y = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])
print(cross_entropy(y, t))

y = np.array([0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0])
print(cross_entropy(y, t))