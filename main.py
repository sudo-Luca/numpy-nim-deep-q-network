"""main.py
Train two DQNs and provide a Tkinter UI for human vs ai1 (AI starts) and human vs ai2 (human starts).
"""
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

def select_action(agent, state, env, epsilon=0.0):
    q = agent.predict(state).copy()
    for a in range(agent.action_size):
        if (a+1) not in env.valid_actions():
            q[a] = -1e9
    return int(np.argmax(q)) + 1


def eval_vs_random(agent, env, games=10):
    wins = 0
    for _ in range(games):
        state = env.reset()
        done = False
        turn = 0
        while not done:
            if turn == 0:
                action = select_action(agent, state, env)
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

    for ep in range(episodes):
        state = env.reset()
        done = False
        turn = 0  # ai1 starts

        while not done:
            if turn == 0:
                action = select_action(ai1, state, env)
                next_state, _, done = env.step(action)
                reward1 = -1.0 if done else 0.0
                reward2 = 1.0 if done else 0.0
                buffer1.push(state, action-1, reward1, next_state, done)
            else:
                action = select_action(ai2, state, env)
                next_state, _, done = env.step(action)
                reward2 = -1.0 if done else 0.0
                reward1 = 1.0 if done else 0.0
                buffer2.push(state, action-1, reward2, next_state, done)

            state = next_state
            turn ^= 1
        if (ep+1) % 100 == 0:
            print(f'Episode {ep+1}/{episodes}')

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

        self.current_mode = None
        self.turn = 0
        self.draw_sticks()

    def draw_sticks(self):
        self.canvas.delete('all')
        cols = 10
        spacing_x = 34
        x0, y0 = 20, 20
        for i in range(self.game_env.stones):
            col = i % cols
            row = i // cols
            x = x0 + col * spacing_x
            y = y0 + row * 28
            self.canvas.create_rectangle(x, y, x+12, y+26, fill='#8B4513', outline='')

    def start_vs_ai1(self):
        self.game_env.reset()
        self.current_mode = 'ai1'
        self.turn = 1
        self.status.config(text='AI1 starts')
        self.draw_sticks()
        self.root.after(300, self.ai_move)

    def start_vs_ai2(self):
        self.game_env.reset()
        self.current_mode = 'ai2'
        self.turn = 0
        self.status.config(text='You start (vs AI2)')
        self.draw_sticks()

    def human_move(self, take):
        if self.turn != 0 or take not in self.game_env.valid_actions():
            return
        self.game_env.step(take)
        self.draw_sticks()
        if self.game_env.stones == 0:
            messagebox.showinfo('Result', 'You took the last stick — You lose!')
            return
        self.turn = 1
        self.status.config(text='AI turn')
        self.root.after(300, self.ai_move)

    def ai_move(self):
        if self.turn != 1:
            return
        agent = self.ai1 if self.current_mode == 'ai1' else self.ai2
        state = self.game_env._get_state()
        action = select_action(agent, state, self.game_env)  # deterministic
        self.game_env.step(action)
        self.draw_sticks()
        if self.game_env.stones == 0:
            messagebox.showinfo('Result', 'AI took the last stick — You lose!' if self.current_mode=='ai1' else 'AI took the last stick — AI loses!')
            return
        self.turn = 0
        self.status.config(text='Your turn')

    def show_losses(self):
        global losses_ai1, losses_ai2
        if losses_ai1 is None or losses_ai2 is None:
            messagebox.showinfo('Losses', 'No loss data available')
            return
        plt.figure()
        plt.plot(losses_ai1, label='ai1')
        plt.plot(losses_ai2, label='ai2')
        plt.legend()
        plt.title('Training losses')
        plt.show()


if __name__ == '__main__':
    h1 = (256, 256, 256, 256, 256)
    h2 = (256, 256, 256, 256, 256)

    env = Nim(starting_stones=15, max_take=3)
    ai1 = DQN(state_size=1, action_size=env.max_take, hidden_sizes=h1, lr=1e-3, gamma=0.99)
    ai2 = DQN(state_size=1, action_size=env.max_take, hidden_sizes=h2, lr=1e-3, gamma=0.99)

    print('Training AI agents...')
    losses1, losses2 = train(ai1, ai2, env, episodes=int(input("N epochs --> ")))
    print('Training finished')

    wins = eval_vs_random(ai1, env, games=10)
    print(f'ai1 won {wins}/10 vs random')

    if wins > 6:
        print('ai1 passed threshold — launching GUI')
        root = tk.Tk()
        root.title('Nim: Human vs AI')
        gui = NimGUI(root, ai1, ai2, env)
        root.mainloop()
    else:
        print('ai1 did not pass threshold. Increase training or change architectures.')
