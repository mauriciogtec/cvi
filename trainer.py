import os
import numpy as np
import torch
from cvi import CVI
from buffers import ReplayBufferWithGoals
from envs import ContinuousGridMDP
import matplotlib.pyplot as plt
from matplotlib import animation
from omegaconf import DictConfig
import logging


# turn off annoying messages while saving matplotlib video
logger = logging.getLogger("matplotlib")
logger.setLevel(logging.ERROR)


class Trainer:
    def __init__(
        self,
        train_env: ContinuousGridMDP,
        test_env: ContinuousGridMDP,
        agent: CVI,
        buffer: ReplayBufferWithGoals,
        cfg: DictConfig,
    ) -> None:
        self.buffer = buffer
        self.train_env = train_env
        self.test_env = test_env
        self.state, self.goal = train_env.reset()
        self.agent = agent
        self.cfg = cfg

    def collect(self, random: bool = False) -> None:
        env = self.train_env
        self.agent.eval()
        for step in range(self.cfg.collect_steps):
            s = self.state
            g = env.goal
            a = self.agent.select_action(s, g, greedy=False)
            s1, r, d, _ = env.step(a)
            self.buffer.add(s, a, r, s1, g, d)
            if d:
                self.state, self.goal = env.reset()
            else:
                self.state = s1

    def eval(self) -> None:
        env = self.test_env
        s, g = env.reset()
        states = [s]
        goals = [g]
        self.agent.eval()  # eval model for training
        num_episodes = 0
        num_success = 0.0
        for step in range(self.cfg.eval_steps):
            a = self.agent.select_action(s, env.goal, greedy=True)
            s, r, d, _ = env.step(a)
            if d:
                num_success += r
                num_episodes += 1
                s, g = env.reset()
            states.append(s)
            goals.append(g)
        metrics = dict(av_success=np.round(num_success / num_episodes, 2))
        return metrics, states, goals

    def train_model(self) -> None:
        self.agent.train()
        batch_size = self.cfg.batch_size
        num_batches = self.cfg.num_batches
        dev = self.cfg.device

        model_loss = 0.0
        for b in range(num_batches):
            s, a, r, s1, g, _ = self.buffer.sample(batch_size)
            s_ = torch.FloatTensor(s).to(dev)
            a_ = torch.FloatTensor(a).to(dev)
            s1_ = torch.FloatTensor(s1).to(dev)
            ls = self.agent.update_model(s_, a_, s1_)
            model_loss += ls["model"] / num_batches
        metrics = dict(model=np.round(model_loss, 4))

        return metrics

    def train_value(self) -> None:
        # 0. get training data
        self.agent.train()
        max_samples = self.cfg.max_train_samples
        s, a, r, s1, g, _ = self.buffer.sample_all(max_samples)

        # 1. update state value
        v_eval, *_ = self.value_grid()
        deltas = []
        for _ in range(self.cfg.max_value_iters):
            self.agent.update_state_value(s, r, s1, g)
            v_eval_new, *_ = self.value_grid()
            error = np.mean(np.abs(v_eval - v_eval_new))
            deltas.append(error)
            v_eval = v_eval_new
            if error < self.cfg.vi_tol:
                break
        self.vi_convergence = deltas  # save for plotting later
        metrics = dict(num_vi_iters=len(deltas))

        # 2. update action value
        self.agent.update_action_value(s, a, s1, g)

        return metrics

    def goal_based_rollouts(self, s: np.array, horizon: int) -> tuple:
        tmp_empty = [], [], [], [], [], []
        states, actions, rewards, states_tp1, goals, dn = tmp_empty
        s = s.copy()
        B = s.shape[0]
        g_actual = np.random.uniform(size=s.shape)
        for h in range(horizon):
            states.append(s)
            a = self.agent.select_action(s, g_actual, greedy=False)
            g = np.random.uniform(size=s.shape)
            s_ = torch.FloatTensor(s).to(self.cfg.device)
            a_ = torch.FloatTensor(a).to(self.cfg.device)
            with torch.no_grad():
                s1, _ = self.agent.transition(s_, a_)
                s1 = s1.cpu().numpy()
            r = [self.train_env.reward(s1[i], g[i]) for i in range(B)]
            d = [(ri == 1.0) for ri in r]
            actions.append(a)
            goals.append(g)
            rewards.append(r)
            states_tp1.append(s1)
            dn.append(d)
            s = s1
        states = np.concatenate(states, 0)
        actions = np.concatenate(actions, 0)
        rewards = np.concatenate(rewards, 0)
        states_tp1 = np.concatenate(states_tp1, 0)
        goals = np.concatenate(goals, 0)
        dn = np.concatenate(dn, 0)
        return states, actions, rewards, states_tp1, goals, dn

    # TODO:
    # separate augmented from real
    def augment_experience(self) -> None:
        n = self.cfg.collect_steps
        t = self.cfg.env.time_limit
        s, a, _, s1, _, _ = self.buffer.sample_most_recent(n)
        M = self.cfg.augment.multiplier

        if self.cfg.augment.HER:
            for i in range(1, n):
                for j in range(i):
                    g_aug = s[i]
                    r_aug = self.env.reward(s1[j], g_aug)
                    dn = r_aug == 1
                    self.buffer.add(s[j], a[j], r_aug, s1[j], g_aug, dn)

        if self.cfg.augment.IER:
            num_samples = (t + 1) * M // 2
            for i in range(num_samples):
                g_aug = np.random.uniform(size=self.agent.state_dim)
                for j in range(n):
                    r_aug = self.train_env.reward(s1[j], g_aug)
                    dn = r_aug == 1.0
                    self.buffer.add(s[j], a[j], r_aug, s1[j], g_aug, dn)

        if self.cfg.augment.rollouts:
            H = self.cfg.augment.rollout_horizon
            batches = (t + 1) * M // (2 * H)
            for _ in range(batches):
                s_, a_, r_, s1_, g_, d_ = self.goal_based_rollouts(s, H)
                for j in range(s_.shape[0]):
                    self.buffer.add(s_[j], a_[j], r_[j], s1_[j], g_[j], d_[j])

    def value_grid(self) -> np.array:
        env = self.train_env
        resol = self.cfg.grid_resolution
        agent = self.agent
        grid_x = np.linspace(env.min_x, env.max_x, num=resol)
        grid_y = np.linspace(env.min_y, env.max_y, num=resol)
        xv, yv = np.meshgrid(grid_x, grid_y)
        g = np.repeat(np.expand_dims(env.goal, 0), resol ** 2, axis=0)
        s = np.stack([xv.flatten(), yv.flatten()], -1)
        s = np.concatenate([s, g], -1)
        v = agent.V.predict(s)
        v = np.reshape(v, (resol, resol))
        return v, env.goal, grid_x, grid_y

    def plot_value_grid(self, file: str) -> None:
        v, g, *_ = self.value_grid()
        plt.figure(figsize=(6, 6))
        plt.imshow(v)
        plt.clim(0, 1)
        plt.colorbar()
        g_rescaled = g * self.cfg.grid_resolution
        plt.scatter(*g_rescaled, s=250, c="red", alpha=0.5)
        plt.title("Value Grid")
        os.makedirs(os.path.dirname(file), exist_ok=True)
        plt.savefig(file)
        plt.close()

    def animate(self, file: str) -> None:
        _, states, goals = self.eval()
        env = self.test_env
        fig = plt.figure(figsize=(5, 5))
        ax = plt.axes(xlim=(env.min_x, env.max_x), ylim=(env.min_y, env.max_y))
        scat1 = ax.scatter([], [], c="blue", s=500, alpha=0.5)
        scat2 = ax.scatter([], [], c="red", s=200, alpha=0.5)
        scat = [scat1, scat2]
        plt.xticks(np.linspace(0.0, 1.0, 10))
        plt.yticks(np.linspace(0.0, 1.0, 10))
        plt.grid(True)
        plt.title("Agent's position")

        def init():
            scat1.set_offsets([])
            scat2.set_offsets([])
            return scat

        def animate(i):
            scat1.set_offsets(np.expand_dims(states[i], 0))
            scat2.set_offsets(np.expand_dims(goals[i], 0))
            return scat

        anim = animation.FuncAnimation(
            fig, animate, init_func=init, frames=100, interval=20, blit=True,
        )
        os.makedirs(os.path.dirname(file), exist_ok=True)
        anim.save(
            file, dpi=75, fps=4, extra_args=["-loglevel", "error"],
        )
        plt.close()

    def plot_vi_convergence(self, file: str) -> None:
        deltas = self.vi_convergence
        plt.figure(figsize=(6, 6))
        plt.plot(deltas)
        plt.title("Value Grid")
        plt.xlabel("Value iterations")
        plt.ylabel("Mean grid absolute difference")
        os.makedirs(os.path.dirname(file), exist_ok=True)
        plt.savefig(file)
        plt.close()
