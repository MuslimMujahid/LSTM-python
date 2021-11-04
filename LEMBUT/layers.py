import numpy as np
from random import uniform
from .functions import ACTIVATION_FUNCTIONS
from .util import TimeseriesDataset


class Dense:
    def __init__(self, units, activation="linear", name="dense", input_shape=None, bias=None):
        self.activation = activation
        self.units = units
        self.bias = bias

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        if (self.bias is not None):
            self.input = np.append(X, np.reshape(
                [self.bias for _ in range(X.shape[0])], (X.shape[0], 1)), axis=1)
        else:
            self.input = X

        input_size = self.input.shape[0 if len(self.input.shape) == 1 else 1]

        if self.W is None:
            self.W = np.zeros([input_size, self.units], dtype=float)

        self.net = np.dot(self.input, self.W)
        self.output = ACTIVATION_FUNCTIONS[self.activation](self.net)
        return self.output


class Cell:
    def __init__(self, n_h, input_shape):
        self.n_h = n_h
        self.batch_size, self.seq_len, self.n_features = input_shape
        self.params = {}

        # forget gate
        self.params["Wf"] = np.random.randn(self.n_h, self.n_h)
        self.params["Uf"] = np.random.randn(self.n_h, self.seq_len)
        self.params["bf"] = np.ones((self.n_h, self.n_features))

        # input gate
        self.params["Wi"] = np.random.randn(self.n_h, self.n_h)
        self.params["Ui"] = np.random.randn(self.n_h, self.seq_len)
        self.params["bi"] = np.ones((self.n_h, self.n_features))

        # cell gate
        self.params["Wc"] = np.random.randn(self.n_h, self.n_h)
        self.params["Uc"] = np.random.randn(self.n_h, self.seq_len)
        self.params["bc"] = np.ones((self.n_h, self.n_features))

        # output gate
        self.params["Wo"] = np.random.randn(self.n_h, self.n_h)
        self.params["Uo"] = np.random.randn(self.n_h, self.seq_len)
        self.params["bo"] = np.ones((self.n_h, self.n_features))

        # output
        self.params["V"] = np.random.randn(1, self.n_h)
        self.params["by"] = np.ones((1, self.n_features))

        # cell and hidden state
        self.params["c"] = np.random.randn(self.n_h, self.n_features)
        self.params["h"] = np.random.randn(self.n_h, self.n_features)

    def forward_step(self, x, prev_c, prev_h):
        # print("x", x.shape)
        # print("Uf", self.params["Uf"].shape)
        # print("Wf", self.params["Wf"].shape)
        # print("bf", self.params["bf"].shape)
        # print("h(t-1)", prev_h.shape)
        # print("c(t-1)", prev_c.shape)
        f = ACTIVATION_FUNCTIONS["sigmoid"](np.dot(
            self.params["Uf"], x) + np.dot(self.params["Wf"], prev_h) + self.params["bf"])
        i = ACTIVATION_FUNCTIONS["sigmoid"](np.dot(
            self.params["Ui"], x) + np.dot(self.params["Wi"], prev_h) + self.params["bi"])
        c_bar = ACTIVATION_FUNCTIONS["tanh"](np.dot(
            self.params["Uc"], x) + np.dot(self.params["Wc"], prev_h) + self.params["bc"])

        self.params["c"] = (f * prev_c) + (i * c_bar)
        o = ACTIVATION_FUNCTIONS["sigmoid"](np.dot(
            self.params["Uo"], x) + np.dot(self.params["Wo"], prev_h) + self.params["bo"])
        self.params["h"] = o * ACTIVATION_FUNCTIONS["tanh"](self.params["c"])

        return self.params["c"], self.params["h"]


class LSTM:
    def __init__(self, n_h, input_shape, output_activation="linear"):
        self.input_shape = input_shape
        self.batch_size, self.seq_len, self.n_features = self.input_shape
        self.n_h = n_h
        self.output_activation = ACTIVATION_FUNCTIONS[output_activation]

    def forward(self, x):
        # Convert to timeseries form
        timeseries_x = TimeseriesDataset(x, self.seq_len)

        # Create cells
        self.cells = [Cell(self.n_h, self.input_shape)
                      for _ in range(timeseries_x.max_index)]

        # Initial cell and hidden state
        c = np.random.randn(self.n_h, self.n_features)
        h = np.random.randn(self.n_h, self.n_features)

        # Loop through all cells
        for idx, cell in enumerate(self.cells):
            c, h = cell.forward_step(timeseries_x.getItem(idx), c, h)

        # get output
        last_cell = self.cells[-1]
        # print("V", last_cell.params["V"].shape)
        # print("h(t)", last_cell.params["h"].shape)
        # print("by", last_cell.params["by"].shape)
        y_hat = self.output_activation(
            np.dot(last_cell.params["V"], last_cell.params["h"]) + last_cell.params["by"])

        return y_hat
