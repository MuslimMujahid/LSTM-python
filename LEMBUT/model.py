import numpy as np
from tabulate import tabulate


class Sequential:
    def __init__(self, layers=[]):
        self.layers = layers
        self.has_run = False

    def __call__(self, x):
        return self.predict(x)

    def add(self, layer):
        self.layers.append(layer)

    def predict(self, x):
        output = x
        for layer in self.layers:
            output = layer.forward(output)

        return output
