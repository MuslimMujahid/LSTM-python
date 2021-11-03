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
        self.params["Uf"] = np.random.randn(self.n_h, self.n_features)
        self.params["bf"] = np.ones((self.n_h, self.seq_len))

        # input gate
        self.params["Wi"] = np.random.randn(self.n_h, self.n_h)
        self.params["Ui"] = np.random.randn(self.n_h, self.n_features)
        self.params["bi"] = np.ones((self.n_h, self.seq_len))

        # cell gate
        self.params["Wc"] = np.random.randn(self.n_h, self.n_h)
        self.params["Uc"] = np.random.randn(self.n_h, self.n_features)
        self.params["bc"] = np.ones((self.n_h, self.seq_len))

        # output gate
        self.params["Wo"] = np.random.randn(self.n_h, self.n_h)
        self.params["Uo"] = np.random.randn(self.n_h, self.n_features)
        self.params["bo"] = np.ones((self.n_h, self.seq_len))

    def forward_step(self, x, prev_c, prev_h):
        f = ACTIVATION_FUNCTIONS["sigmoid"](np.dot(self.params["Uf"], x.transpose(
        )) + np.dot(self.params["Wf"], prev_h) + self.params["bf"])
        i = ACTIVATION_FUNCTIONS["sigmoid"](np.dot(self.params["Ui"], x.transpose(
        )) + np.dot(self.params["Wi"], prev_h) + self.params["bi"])
        c_bar = ACTIVATION_FUNCTIONS["tanh"](np.dot(self.params["Uc"], x.transpose(
        )) + np.dot(self.params["Wc"], prev_h) + self.params["bc"])

        self.params["c"] = (f * prev_c) + (i * c_bar)
        o = ACTIVATION_FUNCTIONS["sigmoid"](np.dot(self.params["Uo"], x.transpose(
        )) + np.dot(self.params["Wo"], prev_h) + self.params["bo"])
        self.params["h"] = o * ACTIVATION_FUNCTIONS["tanh"](self.params["c"])

        return self.params["c"], self.params["h"]


class LSTM:
    def __init__(self, input_shape, n_h=10):
        self.input_shape = input_shape
        self.batch_size, self.seq_len, self.n_features = self.input_shape
        self.n_h = n_h

    def forward(self, x):
        # Convert to timeseries form
        timeseries_x = TimeseriesDataset(x, self.seq_len)

        # Create cells
        self.cells = [Cell(self.n_h, self.input_shape)
                      for _ in range(timeseries_x.max_index)]

        # Initial cell and hidden state
        c = np.random.randn(self.n_h, self.seq_len)
        h = np.random.randn(self.n_h, self.seq_len)

        # Loop through all cells
        for idx, cell in enumerate(self.cells):
            c, h = cell.forward_step(timeseries_x.getItem(idx), c, h)

        return h
