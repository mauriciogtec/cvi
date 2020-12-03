import os
from typing import Optional
import numpy as np
import torch
from cvi import CVI
from buffers import ReplayBufferWithGoals
from envs import ContinuousGridMDP
import matplotlib.pyplot as plt
import matplotlib
from omegaconf import DictConfig
import logging


# turn off annoying messages while saving matplotlib video
logger = logging.getLogger("matplotlib")
logger.setLevel(logging.ERROR)
matplotlib.use("Agg")  # for display-less server


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
        n = self.cfg.collect_steps
        self.agent.eval()
        dones = []  # to compute use the # of baseline HER augments
        for _ in range(n):
            s = self.state
            g = env.goal
            a = self.agent.select_action(s, g, greedy=False)
            s1, r, d, _ = env.step(a)
            dones.append(d)
            self.buffer.add(s, a, r, s1, g, d)
            if d:
                self.state, self.goal = env.reset()
            else:
                self.state = s1

        # compute how many HER augments and store
        trajs = []
        curr = []
        t = 0  # possible augments in HER
        for i in range(n):
            if i < n - 1 and not dones[i]:
                curr.append(i)
            else:
                if len(curr) > 1:
                    trajs.append(curr)
                    q = len(curr)
                    t += q * (q + 1) // 2
                curr = [i]
        self.trajectories = trajs
        self.her_possible_augments = t

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
        n = batch_size * num_batches
        s, a, r, s1, g, _ = self.buffer.sample_most_recent(n)
        for b in range(num_batches):
            ix = range(b * batch_size, (b + 1) * batch_size)
            s_ = torch.FloatTensor(s[ix]).to(dev)
            a_ = torch.FloatTensor(a[ix]).to(dev)
            s1_ = torch.FloatTensor(s1[ix]).to(dev)
            ls = self.agent.update_model(s_, a_, s1_)
            model_loss += ls["model"] / num_batches
        metrics = dict(model=np.round(model_loss, 4))

        return metrics

    def train_value(self) -> None:
        # 0. get training data
        self.agent.train()
        max_samples = self.cfg.max_train_samples

        # 1. update state value
        v_eval, *_ = self.value_grid()
        deltas = []

        s, a, r, s1, g, _ = self.buffer.sample_all(max_samples)

        M = self.cfg.augment.multiplier
        H = self.cfg.augment.rollout_horizon
        num_rollouts = max(1, M // H)
        if self.cfg.augment.rollouts:
            root = s.copy()
            s, a, r, s1, g = [], [], [], [], []
            for _ in range(num_rollouts):
                s_, a_, _, s1_, _, _ = self.goal_based_rollouts(root, H)
                for i in range(1, H):
                    if np.random.rand() < self.cfg.augment.random_goal_prob:
                        g_aug = np.random.uniform(size=root.shape)
                    else:
                        g_aug = s_[i]
                    for j in range(i):
                        r_aug = [
                            self.train_env.reward(ss, gg)
                            for ss, gg in zip(s1_[j], g_aug)
                        ]
                        s.append(s_[j])
                        a.append(a_[j])
                        r.append(r_aug)
                        s1.append(s1_[j])
                        g.append(g_aug)
            s = np.concatenate(s, 0)
            a = np.concatenate(a, 0)
            r = np.concatenate(r, 0)
            s1 = np.concatenate(s1, 0)
            g = np.concatenate(g, 0)

        for _ in range(self.cfg.max_value_iters):
            self.agent.update_state_value(s, r, s1, g)
            v_eval_new, *_ = self.value_grid()
            error = np.mean(np.abs(v_eval - v_eval_new))
            deltas.append(error)
            v_eval = v_eval_new
            if error < self.cfg.vi_tol:  # success
                break
            elif self.cfg.resample_value_batch:
                raise NotImplementedError
                # s, a, r, s1, g, _ = self.buffer.sample_all(max_samples)
                # root = s.copy()
                # for _ in range(num_rollouts):
                #     s_, a_, r_, s1_, g_, _ = self.goal_based_rollouts(root, H)
                #     s = np.concatenate([s, s_], 0)
                #     a = np.concatenate([a, a_], 0)
                #     r = np.concatenate([r, r_], 0)
                #     s1 = np.concatenate([s1, s1_], 0)
                #     g = np.concatenate([g, g_], 0)

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
        for _ in range(horizon):
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
        states = np.stack(states, 0)
        actions = np.stack(actions, 0)
        rewards = np.stack(rewards, 0)
        states_tp1 = np.stack(states_tp1, 0)
        goals = np.stack(goals, 0)
        dn = np.stack(dn, 0)
        return states, actions, rewards, states_tp1, goals, dn

    # TODO:
    # separate augmented from real
    def augment_buffer(self) -> None:
        n = self.cfg.collect_steps
        s, a, _, s1, _, dn = self.buffer.sample_most_recent(n)

        trajs = self.trajectories
        if self.cfg.augment.HER:
            for tau in trajs:
                for i in range(1, len(tau)):
                    g_aug = s[tau[i]]
                    for j in tau[:i]:
                        r_aug = self.train_env.reward(s1[j], g_aug)
                        dn = r_aug == 1
                        self.buffer.add(s[j], a[j], r_aug, s1[j], g_aug, dn)

        M = self.cfg.augment.multiplier
        B = s.shape[0]
        num_samples = (M * self.her_possible_augments) // B
        if self.cfg.augment.IER:
            for i in range(num_samples):
                g_aug = np.random.uniform(size=self.agent.state_dim)
                for j in range(n):
                    r_aug = self.train_env.reward(s1[j], g_aug)
                    dn = r_aug == 1.0
                    self.buffer.add(s[j], a[j], r_aug, s1[j], g_aug, dn)

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

        anim = matplotlib.animation.FuncAnimation(
            fig, animate, init_func=init, frames=60, interval=20, blit=True,
        )
        os.makedirs(os.path.dirname(file), exist_ok=True)
        anim.save(
            file, dpi=75, fps=6, extra_args=["-loglevel", "error"],
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
