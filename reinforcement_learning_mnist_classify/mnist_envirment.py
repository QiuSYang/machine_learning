"""
# 创建mnist分类器的环境
"""
import os
import logging
import random

_logger = logging.getLogger(__name__)


class MnistEnviroment(object):
    """参考gym的环境构建"""
    def __init__(self, x, y):
        self.train_x = x
        self.train_y = y
        self.current_index = self._sample_index()
        self.action_space = len(set(y)) - 1

    def reset(self):
        obs, _ = self.step(-1)

        return obs

    def step(self, action):
        """
        :param action 0-9 categori, -1 : start and no reward
        :return: next_state(image), reward
        """
        if action == -1:
            _c_index = self.current_index
            self.current_index = self._sample_index()

            return self.train_x[_c_index], 0

        reward_value = self.reward(action)
        self.current_index = self._sample_index()

        return self.train_x[self.current_index], reward_value

    def reward(self, action):
        c = self.train_y[self.current_index]

        return 1 if c == action else -1

    def sample_actions(self):
        return random.randint(0, self.action_space)

    def _sample_index(self):
        return random.randint(0, len(self.train_y) - 1)
