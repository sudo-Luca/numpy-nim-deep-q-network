"""main.py
Train two DQNs and provide a Tkinter UI for human vs ai1 (AI starts) and human vs ai2 (human starts).
Run: python main.py --hidden1 128,64 --hidden2 64,32 --episodes 10000
"""
import argparse
import numpy as np
import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
from deepQNetwork import DQN, ReplayBuffer
from game import Nim

# Global losses (filled after training)
losses_ai1 = None
losses_ai2 = None

# helpers

def parse_hidden(s):
    if not s:
        return (64,64)
    parts = [p.strip() for p in s.split(',') if p.strip()]
    return tuple(int(p) for p in parts)

def select_action(agent, state, env, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice(env.valid_actions())
    q = agent.predict(state)
    q = q.copy()
    for a in range(agent.action_size):
        if (a+1) not in env.valid_actions():
            q[a] = -1e9
    return int(np.argmax(q)) + 1

def eval_vs_random(agent, env, games=10):
    wins = 0
    for _ in range(games):
        state = env.reset()
        done = False
        turn = 0  # agent first
        while not done:
            if turn == 0:
                action = select_action(agent, state, env, 0.0)
            else:
                action = np.random.choice(env.valid_actions())
            state, _, done = env.step(action)
            turn ^= 1
        if turn == 1:
            wins += 1
    return wins

def train(ai1, ai2, env, episodes=10000, batch_size=64, lr=1e-3, gamma=0.99):
    global losses_ai1, losses_ai2

    buffer1 = ReplayBuffer(20000)
    buffer2 = ReplayBuffer(20000)

    target1 = DQN(ai1.state_size, ai1.action_size, hidden_sizes=ai1.hidden_sizes, lr=lr, gamma=gamma)
    target2 = DQN(ai2.state_size, ai2.action_size, hidden_sizes=ai2.hidden_sizes, lr=lr, gamma=gamma)

    for i in range(len(ai1.weights)):
        target1.weights[i] = ai1.weights[i].copy()
        target1.biases[i] = ai1.biases[i].copy()
    for i in range(len(ai2.weights)):
        target2.weights[i] = ai2.weights[i].copy()
        target2.biases[i] = ai2.biases[i].copy()

    losses1, losses2 = [], []

    for ep in range(1, episodes+1):
        state = env.reset()
        done = False
        turn = 0  # ai1 starts

        while not done:
            if turn == 0:
                action = select_action(ai1, state, env, 0.1)
                next_state, _, done = env.step(action)
                reward = 1.0 if done else 0.0
                buffer1.push(state, action-1, reward, next_state, done)
            else:
                action = select_action(ai2, state, env, 0.1)
                next_state, _, done = env.step(action)
                reward = 1.0 if done else 0.0
                buffer2.push(state, action-1, reward, next_state, done)

            state = next_state
            turn ^= 1

        if len(buffer1) >= batch_size:
            s, a, r, ns, d = buffer1.sample(batch_size)
            q_next = np.max(target1.forward(ns), axis=1)
            target = ai1.forward(s)
            for i in range(batch_size):
                target[i, a[i]] = r[i] + (0 if d[i] else gamma * q_next[i])
            loss = ai1.update(s, target)
            losses1.append(float(loss))
            target1.soft_update(ai1, tau=0.1)

        if len(buffer2) >= batch_size:
            s, a, r, ns, d = buffer2.sample(batch_size)
            q_next = np.max(target2.forward(ns), axis=1)
            target = ai2.forward(s)
            for i in range(batch_size):
                target[i, a[i]] = r[i] + (0 if d[i] else gamma * q_next[i])
            loss = ai2.update(s, target)
            losses2.append(float(loss))
            target2.soft_update(ai2, tau=0.1)
        if ep % 100 == 0:
            print(f'Episode {ep}/{episodes} — Losses: AI1={np.mean(losses1[-100:]):.4f}, AI2={np.mean(losses2[-100:]):.4f}')

    losses_ai1 = losses1
    losses_ai2 = losses2
    return losses1, losses2

# GUI
class NimGUI:
    def __init__(self, root, ai1, ai2, env):
        self.root = root
        self.ai1 = ai1
        self.ai2 = ai2
        self.env = env
        self.game_env = Nim(env.starting_stones, env.max_take)

        self.canvas = tk.Canvas(root, width=500, height=240, bg='white')
        self.canvas.pack(padx=8, pady=8)

        controls = tk.Frame(root)
        controls.pack()
        tk.Button(controls, text='Play vs AI1 (AI starts)', command=self.start_vs_ai1).pack(side='left', padx=4)
        tk.Button(controls, text='Play vs AI2 (You start)', command=self.start_vs_ai2).pack(side='left', padx=4)
        tk.Button(controls, text='Show losses', command=self.show_losses).pack(side='left', padx=4)

        moves = tk.Frame(root)
        moves.pack(pady=6)
        for i in range(1, self.env.max_take+1):
            tk.Button(moves, text=f'Take {i}', command=lambda a=i: self.human_move(a)).pack(side='left', padx=6)

        self.status = tk.Label(root, text='')
        self.status.pack(pady=6)

        self.current_mode = None  # 'ai1' or 'ai2'
        self.turn = 0
        self.draw_sticks()

    def draw_sticks(self):
        self.canvas.delete('all')
        cols = 10
        spacing_x = 34
        x0 = 20
        y0 = 20
        for i in range(self.game_env.stones):
            col = i % cols
            row = i // cols
            x = x0 + col * spacing_x
            y = y0 + row * 28
            self.canvas.create_rectangle(x, y, x+12, y+26, fill='#8B4513', outline='')

    def start_vs_ai1(self):
        self.game_env.reset()
        self.current_mode = 'ai1'
        self.turn = 1  # AI starts
        self.status.config(text='AI1 starts')
        self.draw_sticks()
        self.root.after(300, self.ai_move)

    def start_vs_ai2(self):
        self.game_env.reset()
        self.current_mode = 'ai2'
        self.turn = 0  # human starts
        self.status.config(text='You start (vs AI2)')
        self.draw_sticks()

    def human_move(self, take):
        if self.turn != 0:
            return
        if take not in self.game_env.valid_actions():
            self.status.config(text='Invalid move')
            return
        self.game_env.step(take)
        self.draw_sticks()
        if self.game_env.stones == 0:
            messagebox.showinfo('Result', 'You took the last stick — You win!')
            return
        self.turn = 1
        self.status.config(text='AI turn')
        self.root.after(300, self.ai_move)

    def ai_move(self):
        if self.turn != 1:
            return
        agent = self.ai1 if self.current_mode == 'ai1' else self.ai2
        state = self.game_env._get_state()
        action = select_action(agent, state, self.game_env, epsilon=0.0)
        self.game_env.step(action)
        self.draw_sticks()
        if self.game_env.stones == 0:
            messagebox.showinfo('Result', 'AI took the last stick — AI wins!')
            return
        self.turn = 0
        self.status.config(text='Your turn')

    def show_losses(self):
        global losses_ai1, losses_ai2
        if losses_ai1 is None or losses_ai2 is None:
            messagebox.showinfo('Losses', 'No loss data available (train first)')
            return
        plt.figure()
        plt.plot(losses_ai1, label='ai1')
        plt.plot(losses_ai2, label='ai2')
        plt.legend()
        plt.title('Training losses')
        plt.show()

# Main
if __name__ == '__main__':
    # Fixed architectures (edit here)
    h1 = (64, 128, 128, 128, 128, 128, 64)
    h2 = (64, 128, 128, 256, 32, 64)

    env = Nim(starting_stones=15, max_take=3)
    ai1 = DQN(state_size=1, action_size=env.max_take, hidden_sizes=h1, lr=1e-3, gamma=0.99)
    ai2 = DQN(state_size=1, action_size=env.max_take, hidden_sizes=h2, lr=1e-3, gamma=0.99)

    print('Training AI agents...')
    losses1, losses2 = train(ai1, ai2, env, episodes=int(input("N epochs --> ")))
    print('Training finished')

    wins = eval_vs_random(ai1, env, games=10)
    print(f'ai1 won {wins}/10 vs random')

    if wins >= 5:
        print('ai1 passed threshold — launching GUI')
        root = tk.Tk()
        root.title('Nim: Human vs AI')
        gui = NimGUI(root, ai1, ai2, env)
        root.mainloop()
    else:
        print('ai1 did not pass threshold. Increase training or change architectures.')
