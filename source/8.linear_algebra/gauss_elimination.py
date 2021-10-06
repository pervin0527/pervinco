import numpy as np

A = np.array([[2,1,1],[4,-6,0],[-2,7,2]])

E21 = np.array([[1,0,0],[-2,1,0],[0,0,1]])
A_prime = np.dot(E21, A)

E31 = np.array([[1,0,0],[0,1,0],[1,0,1]])
A_prime = np.dot(E31, A_prime)

E32 = np.array([[1,0,0], [0,1,0], [0,1,1]])
U = np.dot(E32, A_prime)

E21_i = np.array([[1,0,0],[2,1,0],[0,0,1]])
E31_i = np.array([[1,0,0],[0,1,0],[-1,0,1]])
E32_i = np.array([[1,0,0], [0,1,0], [0,-1,1]])

L = np.dot(E31_i, E32_i)
L = np.dot(E21_i, L)

if np.all(np.dot(L, U) == A):
    print("Same")
    print(np.dot(L, U))

else:
    print("Not Same")