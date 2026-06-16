"""
buffer.py — ReplayBuffer (MBPO용, DOB 필드 없음)
real_buffer와 model_buffer를 분리된 인스턴스로 운영.
"""
import numpy as np


class ReplayBuffer:
    def __init__(self, buffer_size: int, num_obs: int = 11, num_act: int = 3):
        self.size   = buffer_size
        self.index  = 0
        self.length = 0

        self.obs      = np.zeros((buffer_size, num_obs), dtype=np.float32)
        self.next_obs = np.zeros((buffer_size, num_obs), dtype=np.float32)
        self.act      = np.zeros((buffer_size, num_act), dtype=np.float32)
        self.rew      = np.zeros(buffer_size,             dtype=np.float32)
        self.done     = np.zeros(buffer_size,             dtype=bool)

    def store(self, obs, act, next_obs, rew, done):
        idx = self.index
        self.obs[idx]      = obs
        self.act[idx]      = act
        self.next_obs[idx] = next_obs
        self.rew[idx]      = rew
        self.done[idx]     = done
        self.index  = (self.index + 1) % self.size
        self.length = min(self.length + 1, self.size)

    def store_batch(self, obs_b, act_b, next_obs_b, rew_b, done_b):
        n       = len(obs_b)
        indices = np.arange(self.index, self.index + n) % self.size
        self.obs[indices]      = obs_b
        self.act[indices]      = act_b
        self.next_obs[indices] = next_obs_b
        self.rew[indices]      = rew_b
        self.done[indices]     = done_b
        self.index  = (self.index + n) % self.size
        self.length = min(self.length + n, self.size)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.length, size=batch_size)
        return (self.obs[idx], self.act[idx], self.next_obs[idx],
                self.rew[idx], self.done[idx])

    def get_all(self):
        return (self.obs[:self.length], self.act[:self.length],
                self.next_obs[:self.length], self.rew[:self.length],
                self.done[:self.length])
