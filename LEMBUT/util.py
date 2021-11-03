import pandas as pd
import numpy as np


class TimeseriesDataset:
    def __init__(self, dataset, seq_len):
        self.dataset = np.array(dataset)
        self.seq_len = seq_len
        self.max_index = len(dataset) - seq_len + 1

    def getItem(self, i):
        return self.dataset[i:i+self.seq_len, :]
