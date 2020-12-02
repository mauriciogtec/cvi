"""
This file implements regular ReplayBuffer, Prioritized Replay Buffer and Trajectory Replay Buffer
"""
import random
from typing import List, Union, Optional

import numpy as np
import torch


class ReplayBufferWithGoals(object):
    """
    Replay buffer to store transitions and get them at random
    """

    def __init__(self, size: int):
        """
        Implements a ring buffer (FIFO).

        :param size: (int)  Max number of transitions to store in the buffer. When the buffer overflows the old
            memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self) -> int:
        return len(self._storage)

    @property
    def storage(self):
        """[(Union[np.ndarray, int], Union[np.ndarray, int], float, Union[np.ndarray, int], bool)]:
         content of the replay buffer"""
        return self._storage

    @property
    def buffer_size(self) -> int:
        """float: Max capacity of the buffer"""
        return self._maxsize

    def can_sample(self, n_samples: int) -> bool:
        """
        Check if n_samples samples can be sampled
        from the buffer.

        :param n_samples: (int)
        :return: (bool)
        """
        return len(self) >= n_samples

    def is_full(self) -> int:
        """
        Check whether the replay buffer is full or not.

        :return: (bool)
        """
        return len(self) == self.buffer_size

    def add(self, obs_t, action, reward, obs_tp1, goal, done):
        """
        add a new transition to the buffer

        :param obs_t: (Union[np.ndarray, int]) the last observation
        :param action: (Union[np.ndarray, int]) the action
        :param reward: (float) the reward of the transition
        :param obs_tp1: (Union[np.ndarray, int]) the current observation
        :param done: (bool) is the episode done
        """
        data = (obs_t, action, reward, obs_tp1, goal, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes: Union[List[int], np.ndarray]):
        tmp = [], [], [], [], [], []
        obses_t, actions, rewards, obses_tp1, goals, dones = tmp
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, goal, done = data
            obses_t.append(obs_t)
            actions.append(action)
            rewards.append(reward)
            obses_tp1.append(obs_tp1)
            goals.append(goal)
            dones.append(done)

        obses_t = np.array(obses_t, copy=False)
        actions = np.array(actions, copy=False)
        obses_tp1 = np.array(obses_tp1, copy=False)
        rewards = np.array(rewards, copy=False)
        goals = np.array(goals, copy=False)
        dones = np.array(dones, copy=False)
        return (obses_t, actions, rewards, obses_tp1, goals, dones)

    def sample(self, batch_size: int, **_kwargs):
        """
        Sample a batch of experiences.

        :param batch_size: (int) How many transitions to sample.
        :return:
            - obs_batch: (np.ndarray) batch of observations
            - act_batch: (numpy float) batch of actions executed given obs_batch
            - rew_batch: (numpy float) rewards received as results of executing act_batch
            - next_obs_batch: (np.ndarray) next set of observations seen after executing act_batch
            - done_mask: (numpy bool) done_mask[i] = 1 if executing act_batch[i] resulted in the end of an episode
                and 0 otherwise.
        """
        idxes = [
            random.randint(0, len(self._storage) - 1)
            for _ in range(batch_size)
        ]
        return self._encode_sample(idxes)

    def sample_all(self, max_samples: Optional[int] = None):
        idxes = np.random.permutation(len(self))
        if max_samples is not None:
            idxes = idxes[:max_samples]
        return self._encode_sample(idxes)

    def sample_most_recent(self, size: int):
        b = self._next_idx
        N = self._maxsize
        idxes = [i % N for i in range(b - size, b)]
        return self._encode_sample(idxes)
