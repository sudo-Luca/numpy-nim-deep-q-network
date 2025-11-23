"""deepQnetwork.py
DQN and ReplayBuffer implemented with numpy only.
Hidden layer architecture is flexible (provide tuple/list of integers).
"""
import numpy as np
import pickle

class ReplayBuffer:
    def __init__(self, capacity=20000):
        self.capacity = int(capacity)
        self.buffer = []
        self.pos = 0

    def push(self, state, action, reward, next_state, done):
        item = (state, action, reward, next_state, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(item)
        else:
            self.buffer[self.pos] = item
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        idx = np.random.choice(len(self.buffer), int(batch_size), replace=False)
        states, actions, rewards, next_states, dones = zip(*(self.buffer[i] for i in idx))
        return (np.array(states, dtype=np.float32),
                np.array(actions, dtype=np.int32),
                np.array(rewards, dtype=np.float32),
                np.array(next_states, dtype=np.float32),
                np.array(dones, dtype=np.bool_))

    def __len__(self):
        return len(self.buffer)

class DQN:
    def __init__(self, state_size, action_size, hidden_sizes=(64,64), lr=1e-3, gamma=0.99):
        self.state_size = int(state_size)
        self.action_size = int(action_size)
        self.hidden_sizes = tuple(int(h) for h in hidden_sizes)
        self.lr = float(lr)
        self.gamma = float(gamma)

        sizes = [self.state_size] + list(self.hidden_sizes) + [self.action_size]
        self.weights = []
        self.biases = []
        for i in range(len(sizes)-1):
            w = np.random.randn(sizes[i], sizes[i+1]) * np.sqrt(2.0 / max(1, sizes[i]))
            b = np.zeros(sizes[i+1], dtype=np.float32)
            self.weights.append(w.astype(np.float32))
            self.biases.append(b
                                  )

    def forward(self, x):
        a = x.astype(np.float32)
        for i in range(len(self.weights)-1):
            a = np.dot(a, self.weights[i]) + self.biases[i]
            a = np.tanh(a)
        out = np.dot(a, self.weights[-1]) + self.biases[-1]
        return out

    def predict(self, state):
        s = np.atleast_2d(state).astype(np.float32)
        return self.forward(s)[0]

    def update(self, states, targets):
        # forward pass collecting activations
        batch = states.shape[0]
        activations = [states.astype(np.float32)]
        a = activations[0]
        for i in range(len(self.weights)-1):
            a = np.dot(a, self.weights[i]) + self.biases[i]
            a = np.tanh(a)
            activations.append(a)
        preds = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]

        # mean squared error gradient on outputs
        grad_out = (preds - targets) * (2.0 / batch)

        # gradients for last layer
        grad_w = np.dot(activations[-1].T, grad_out)
        grad_b = np.sum(grad_out, axis=0)
        self.weights[-1] -= self.lr * grad_w
        self.biases[-1] -= self.lr * grad_b

        # backpropagate through earlier layers
        grad = np.dot(grad_out, self.weights[-1].T)
        for i in range(len(self.weights)-2, -1, -1):
            da = grad * (1 - activations[i+1]**2)  # tanh'
            grad_w_i = np.dot(activations[i].T, da)
            grad_b_i = np.sum(da, axis=0)
            self.weights[i] -= self.lr * grad_w_i
            self.biases[i] -= self.lr * grad_b_i
            if i > 0:
                grad = np.dot(da, self.weights[i].T)

        loss = np.mean((preds - targets)**2)
        return loss

    def soft_update(self, source, tau=0.01):
        for i in range(len(self.weights)):
            self.weights[i] = (1.0 - tau) * self.weights[i] + tau * source.weights[i]
            self.biases[i] = (1.0 - tau) * self.biases[i] + tau * source.biases[i]

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump((self.weights, self.biases), f)

    def load(self, path):
        with open(path, 'rb') as f:
            self.weights, self.biases = pickle.load(f)