# evolved from forward_propagation.py

import numpy as np

class Sigmoid:
    def __init__(self):
        # grads => gradient list
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        return 1 / (1 + np.exp(-x))

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx

# between sigmoid and sigmoid
class Affine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x):
        W, b = self.params
        out = np.matmul(x, W) + b
        self.x = x
        return out

    def backward(self, dout):
        W, b = self.params
        dx = np.matmul(dout, W.T)
        dw = np.matmul(self.x.T, dout)
        db = np.sum(dout, axis=0)

        self.grads[0][...] = dw
        self.grads[1][...] = db
        return db

# flow
# x -> Affine -> Sigmoid -> Affine -> s

class TowLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        I, H, O = input_size, hidden_size, output_size

        # params init
        W1 = np.random.randn(I, H)
        b1 = np.random.randn(H)
        W2 = np.random.randn(H, O)
        b2 = np.random.randn(O)

        # make layer
        self.layers = [
            Affine(W1, b1),
            Sigmoid(),
            Affine(W2, b2)
        ]

        # collect all W
        self.params = []
        for layer in self.layers:
            self.params = layer.params

    def predict(self, x):
        for layer in self.layers:
            # previous output = present input
            x = layer.forward(x)
        return x

x = np.random.randn(10 ,2)
model = TowLayerNet(2, 4, 3)
s = model.predict(x)
print(s)

# last W, b check
print(model.params)

