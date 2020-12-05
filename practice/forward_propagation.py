import numpy as np

class Sigmoid:
    def __init__(self):
        self.params = []

    def forward(self, x):
        return 1 / (1 + np.exp(-x))

class Affine:
    def __init__(self, W, b):
        self.params = [W, b]

    def forward(self, x):
        W, b = self.params
        return np.matmul(x, W) + b

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

