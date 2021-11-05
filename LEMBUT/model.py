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
        self.has_run = True
        output = x
        for layer in self.layers:
            output = layer.forward(output)

        return output

    def summary(self):
        table = []
        heads = ["Layer (type)", "Output Shape", "Params"]
        prev_output_shape = None
        total_param = 0
        for idx, layer in enumerate(self.layers):
            name = layer.type
            output_Shape = None
            param = None
            if (idx == 0):
                if layer.input_shape == None:
                    continue
                if (name == 'Dense'):
                    output_shape = f'(None, {layer.units})'
                    param = layer.input_shape[0] * layer.units
                    prev_output_shape = (layer.units, )
                elif (name == 'LSTM'):
                    output_shape = f'({layer.input_shape[0]}, {layer.input_shape[1]}, {layer.n_h})'
                    param = 4*((layer.n_h+layer.input_shape[2])*layer.n_h+layer.n_h)
                    prev_output_shape = (layer.n_h, )
            else:
                if (name == 'Dense'):
                    output_shape = f'(None, {layer.units})'
                    param = (prev_output_shape[0]+1) * layer.units
                    prev_output_shape = (layer.units, )
                elif (name == 'LSTM'):
                    output_shape = f'({layer.input_shape[0]}, {layer.n_h})'
                    param = 4*((layer.n_h+prev_output_shape[0]+1)*layer.n_h+layer.n_h)
                    prev_output_shape = (layer.n_h, )

            table.append([f'{layer.name} ({name})', output_shape, param])
            total_param += param

        print()
        print(tabulate(table, headers=heads, tablefmt="github"))
        print()
        print("Total params:", total_param)
        print("Trainable params:", total_param)
        print("Non-Trainable params:", 0)