import numpy as np
import matplotlib.pyplot as plt

data = open("lovecraft.txt").read().lower()

chars = set(data)
vocab_size = len(chars)
print("Data has %d characters and %d unique chars" % (len(data), vocab_size))

char_to_idx = {w: i for i, w in enumerate(chars)}
idx_to_char = {i: w for i, w in enumerate(chars)}

class LSTM:
    def __init__(self, char_to_idx, idx_to_char, vocab_size, n_h=100, seq_len=25,
                 epochs=10, lr=0.01, beta1=0.9, beta2=0.999) -> None:
        self.char_to_idx = char_to_idx
        self.idx_to_char = idx_to_char
        self.vocab_size = vocab_size
        self.n_h = n_h
        self.seq_len = seq_len
        self.epochs = epochs
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2

        self.params = {}
        # Xavier initialization
        std = (1.0/np.sqrt(self.vocab_size + self.n_h))

        # forget gate
        self.params["Wf"] = np.random.randn(self.n_h, self.n_h + self.vocab_size) * std
        self.params["bf"] = np.ones((self.n_h, 1))

        # input gate
        self.params["Wi"] = np.random.randn(self.n_h, self.n_h + self.vocab_size) * std
        self.params["bi"] = np.ones((self.n_h, 1))

        # cell gate
        self.params["Wc"] = np.random.randn(self.n_h, self.n_h + self.vocab_size) * std
        self.params["bc"] = np.ones((self.n_h, 1))

        # output gate
        self.params["Wo"] = np.random.randn(self.n_h, self.n_h + self.vocab_size) * std
        self.params["bo"] = np.ones((self.n_h, 1))

        # output
        self.params["Wv"] = np.random.randn(self.vocab_size, self.n_h) * (1.0/np.sqrt(self.vocab_size))
        self.params["bv"] = np.zeros((self.vocab_size, 1))

        self.grads = {}
        self.adams_params = {}

        for key in self.params:
            self.grads["d" + key] = np.zeros_like(self.params[key])
            self.adams_params["m" + key] = np.zeros_like(self.params[key])
            self.adams_params["v" + key] = np.zeros_like(self.params[key])
        
        self.smooth_loss = -np.log(1.0 / self.vocab_size) * self.seq_len

    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x)
    
    def clip_grads(self):
        for key in self.grads:
            np.clip(self.grads[key], -5, 5, out=self.grads[key])
        return
    
    def reset_grads(self):
        for key in self.grads:
            self.grads[key].fill(0)
        return
    
    def update_params(self, batch_num):
        for key in self.params:
            self.adams_params["m" + key] = self.adams_params["m" + key] * self.beta1 + (1 - self.beta1)  * self.grads["d" + key]
            self.adams_params["v" + key] = self.adams_params["v" + key] * self.beta2 + (1 - self.beta2) * self.grads["d" + key]**2
            m_corr = self.adams_params["m" + key] / (1 - self.beta1**batch_num)
            v_corr = self.adams_params["v" + key] / (1 - self.beta2**batch_num)
            self.params[key] -= self.lr * m_corr / (np.sqrt(v_corr) + 1e-8)
        return
    
    def forward_step(self, x, h_prev, c_prev):
        z = np.row_stack((h_prev, x))
        f = self.sigmoid(np.dot(self.params["Wf"], z) + self.params["bf"])
        i = self.sigmoid(np.dot(self.params["Wi"], z) + self.params["bi"])
        c_bar = np.tanh(np.dot(self.params["Wc"], z) + self.params["bc"])

        c = f * c_prev + i * c_bar
        o = self.sigmoid(np.dot(self.params["Wo"], z) + self.params["bo"])
        h = o * np.tanh(c)

        v = np.dot(self.params["Wv"], h) + self.params["bv"]
        y_hat = self.softmax(v)
        return y_hat, v, h, o, c, c_bar, i, f, z

    def backward_step(self, y, y_hat, dh_next, dc_next, c_prev, z, f, i, c_bar, c, o, h):
        dv = np.copy(y_hat)
        dv[y] -= 1

        self.grads["dWv"] += np.dot(dv, h.T)
        self.grads["dbv"] += dv

        dh = np.dot(self.params["Wv"].T, dv)
        dh += dh_next

        do = dh * np.tanh(c)
        da_o = do * o * (1 - 0)
        self.grads["dWo"] += np.dot(da_o, z.T)
        self.grads["dBo"] += da_o

        dc = dh * o * (1 - np.tanh(c)**2)
        dc += dc_next

        dc_bar = dc * i
        da_c = dc_bar * (1 - c_bar**2)
        self.grads["dWc"] += np.dot(da_c, z.T)
        self.grads["dbc"] += da_c

        di = dc * c_bar
        da_i = di * i * (1 - i)
        self.grads["dWi"] += np.dot(da_i, z.T)
        self.grads["dbc"] += da_c

        df = dc * c_prev
        da_f = df * f * (1 - f)
        self.grads["dWf"] += np.dot(da_f, z.T)
        self.grads["dBf"] += da_f

        dz = (np.dot(self.params["Wf"].T, da_f)
            + np.dot(self.params["Wi"].T, da_i)
            + np.dot(self.params["Wc"].T, da_c)
            + np.dot(self.params["Wo"].T, da_o)
        )

        dh_prev = dz[:self.n_h, :]
        dc_prev = f * dc
        return dh_prev, dc_prev

    def forward_backward(self, x_batch, y_batch, h_prev, c_prev):
        x, z, f, i, c_bar, c, o, y_hat, v, h = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}

        #init t_-1
        h[-1] = h_prev
        c[-1] = c_prev

        loss = 0
        for t in range(self.seq_len):
            x[t] = np.zeros((self.vocab_size, 1))
            x[t][x_batch[t]] = 1

            y_hat[t], v[t], h[t], o[t], c[t], c_bar[t], i[t], f[t], z[t] = self.forward_step(x[t], h[t-1], c[t-1])
            loss += -np.log(y_hat[t][y_batch[t], 0])

        self.reset_grads()

        dh_next = np.zeros_like(h[0])
        dc_next = np.zeros_like(c[0])

        for t in reversed(range(self.seq_len)):
            dh_next, dc_next = self.backward_step(y_batch[t], y_hat[t], dh_next, dc_next, c[t-1], z[t], f[t], i[t], c_bar[t], c[t], o[t], h[t])
        return loss, h[self.seq_len-1], c[self.seq_len-1]

    def sample(self, h_prev, c_prev, sample_size):
        x = np.zeros((self.vocab_size, 1))
        h = h_prev
        c = c_prev
        sample_string = ""

        for t in range(sample_size):
            y_hat, _, h, _, c, _, _, _, _ = self.forward_step(x, h, c)
            idx = np.random.choice(range(self.vocab_size), p=y_hat.ravel())
            x = np.zeros((self.vocab_size, 1))
            x[idx] = 1
            char = self.idx_to_char[idx]
            sample_string += char

        return sample_string
    
    def train(self, X, verbose=True):
        J = []
        num_batches = len(X)
        X_trimmed = X[: num_batches*self.seq_len]

        for epoch in range(self.epochs):
            h_prev = np.zeros((self.n_h, 1))
            c_prev = np.zeros((self.n_h, 1))

            for j in range(0, len(X_trimmed) - self.seq_len, self.seq_len):
                x_batch = [self.char_to_idx[ch] for ch in X_trimmed[j: j + self.seq_len]]
                y_batch = [self.char_to_idx[ch] for ch in X_trimmed[j + 1: j + self.seq_len + 1]]

                loss, h_prev, c_prev = self.forward_backward(x_batch, y_batch, h_prev, c_prev)
                self.smooth_loss = self.smooth_loss * 0.999 + loss * 0.001
                J.append(self.smooth_loss)

                if epoch == 0 and j == 0:
                    self.gradient
        