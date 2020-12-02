from torch import nn
import numpy as np
from omegaconf import DictConfig
from torch import Tensor
import hydra


class CVI(nn.Module):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.transition = hydra.utils.instantiate(cfg.transition)
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.transition.parameters()
        )
        self.num_action_samples = cfg.num_action_samples
        self.num_nbrs = max(
            cfg.state_value.n_neighbors, cfg.action_value.n_neighbors
        )
        self.state_dim = cfg.state_dim
        self.action_dim = cfg.action_dim
        self.epsilon = cfg.epsilon
        self.beta = cfg.beta
        self.gamma = cfg.gamma
        self.max_delta = cfg.max_delta
        self.reset_value()

    def reset_value(self, fill_zeros: bool = True) -> None:
        self.V = hydra.utils.instantiate(self.cfg.state_value)
        self.Q = hydra.utils.instantiate(self.cfg.action_value)
        # initialize at 0

        if fill_zeros:
            K = self.num_nbrs
            s_ = np.zeros((K, self.state_dim))
            a_ = np.zeros((K, self.action_dim))
            g_ = np.zeros((K, self.state_dim))
            y_ = np.zeros(K)
            self.V.fit(np.concatenate([s_, g_], -1), y_)
            self.Q.fit(np.concatenate([s_, a_, g_], -1), y_)

    def update_state_value(
        self, s: np.array, r: np.array, s1: np.array, g: np.array,
    ) -> None:
        """wraps the knn regression model"""
        X = np.concatenate([s, g], -1)
        X1 = np.concatenate([s1, g], -1)
        v_s = self.beta * self.V.predict(X)
        v_s1 = self.gamma * self.V.predict(X1)
        tgt = np.maximum(r, np.maximum(v_s, v_s1))
        self.reset_value(fill_zeros=False)
        self.V.fit(X, tgt)
        return dict(state_value=np.nan)

    def update_model(
        self, s: Tensor, a: Tensor, s1: Tensor,
    ) -> None:
        """Trains with value equivalence"""
        # dynamics loss
        s1hat, _ = self.transition(s, a)
        loss = (s1 - s1hat).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return dict(model=float(loss))

    def update_action_value(
        self, s: np.array, a: np.array, s1: np.array, g: np.array,
    ) -> None:
        X = np.concatenate([s, a, g], -1)
        tgt = self.V.predict(np.concatenate([s1, g], -1))
        self.Q.fit(X, tgt)
        return dict(state_value=np.nan)

    def select_action(
        self, s: np.array, g: np.array, greedy: bool = False
    ) -> np.array:
        single_env = len(s.shape) == 1
        if single_env:  # single-env case, make batch
            s = np.expand_dims(s, 0)
            g = np.expand_dims(g, 0)
        B = s.shape[0]  # batch size
        N = self.num_action_samples
        d = self.max_delta
        a0 = np.random.uniform(-d, d, (B * N, self.action_dim))
        s0 = np.repeat(s, N, axis=0)
        g0 = np.repeat(g, N, axis=0)
        qvals = self.Q.predict(np.concatenate([s0, a0, g0], -1))
        qvals = np.reshape(qvals, (B, N))

        best_action = np.argmax(qvals, axis=-1).astype(np.int32)
        rand_action = np.random.randint(low=0, high=N, size=B)
        candidates = np.reshape(a0, (B, N, self.action_dim))
        a = np.zeros((B, self.action_dim))
        for b in range(B):
            if np.random.rand() < self.epsilon:
                a[b] = candidates[b, rand_action[b]]
            else:
                a[b] = candidates[b, best_action[b]]

        if single_env:
            a = np.squeeze(a, 0)

        return a
