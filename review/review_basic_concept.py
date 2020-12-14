import numpy as np

# row vector => 1 x N
x = np.array([1, 2, 3])
# result => np.ndarray class
print(x.__class__)

# result => (3,)
print(x.shape)
# result => 1
print(x.ndim)

# shape => N x M
# ndim => N

W = np.array([[1, 2, 3], [4, 5, 6]])
Z = np.array([[1, 2, 3], [4, 5, 6]])

print(Z.ndim)
print(W + Z)
print(W * Z)
print(10 * W)

# N, M x M, W => N, W
Y = np.array([[1, 2], [3, 4], [5, 6]])
print(np.dot(W, Y))
print(np.matmul(W, Y))

new_matrix = np.dot(W, Y)
# need to check after dot
print(new_matrix.shape)

# make fully connected layer
W1 = np.random.randn(2, 4)
b1 = np.random.randn(4)
X1 = np.random.randn(10, 2)
print(np.matmul(X1, W1) + b1)

W2 = np.random.randn(4, 3)
b2 = np.random.randn(3)

# return 0 ~ 1 value
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

h = np.matmul(X1, W1) + b1
a = sigmoid(h)
print(a)

s = np.matmul(a, W2) + b2
print(s.shape)
print(s)



