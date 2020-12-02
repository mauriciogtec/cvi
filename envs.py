"""
grid world mdp class
"""
from typing import Optional, Tuple
import numpy as np


class ContinuousGridMDP:
    """
    Grid world continuous MDP
    """

    def __init__(
        self,
        threshold: float = 0.2,
        start=[1.0, 0.0],
        goal=[0.0, 0.0],
        random_goal: bool = False,
        random_start: bool = False,
        time_limit: int = 30,
        **kwargs
    ):
        assert threshold < 0.5
        self.min_x = 0.0
        self.min_y = 0.0
        self.max_x = 1.0
        self.max_y = 1.0
        self.threshold = threshold

        # ignored if random starts
        self.start_state = np.asarray(start)

        if not random_goal:
            self.goal = np.asarray(goal)
        else:
            self.goal = np.random.uniform(size=2)

        self.random_goal = random_goal
        self.random_start = random_start
        self.time_limit = time_limit
        self.t = 0
        self._done = False

    def reset(self) -> np.ndarray:
        """
        reset state to 0
        :return:
        """
        if self.random_goal:
            self.goal = np.random.uniform(size=2)
        if self.random_start:
            # make sure goal doesn't start on target
            while True:
                self._state = np.random.uniform(size=2)
                if self.reward() < 1.0:
                    break
        else:
            self._state = np.copy(self.start_state)

        self.t = 0
        self.done()
        return self._state, self.goal

    def step(self, act: Tuple[float, float]) -> tuple:
        """
        take action and return next state
        :param act:
        :return:
        """

        if self.t >= self.time_limit:
            print("Time limit exceeded. Reset for new episode")
            raise Exception()

        if self._done:
            print("Episode ended. Reset for new episode")
            raise Exception()

        x, y = self._state
        xnew = np.clip(x + act[0], self.min_x, self.max_x)
        ynew = np.clip(y + act[1], self.min_y, self.max_y)
        self._state = np.asarray([xnew, ynew])
        self.t += 1
        self.done()
        return self._state, self.reward(), self._done, {}

    def done(self):
        """
        check if episode over
        :return:
        """
        if self.t >= self.time_limit:
            self._done = True
        elif np.linalg.norm(self._state - self.goal, 2) < self.threshold:
            self._done = True
        else:
            self._done = False
        return

    def reward(
        self, state: Optional[np.array] = None, goal: Optional[np.array] = None
    ) -> float:
        """
        Reward function
        :return:
        """
        if goal is None:
            goal = self.goal
        if state is None:
            state = self._state
        dist = np.linalg.norm(state - goal, 2)
        return float(dist < self.threshold)
