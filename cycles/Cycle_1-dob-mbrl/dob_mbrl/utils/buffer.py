"""
buffer.py — ReplayBufferDOB
real_buffer와 model_buffer는 분리된 인스턴스로 운영.
dob_dim: DOB 차원 (MountainCar = 1, CartPole = 2)
"""
import numpy as np


class ReplayBufferDOB:
    def __init__(self, buffer_size: int, num_obs: int = 2,
                 num_act: int = 1, dob_dim: int = 1):
        self.size   = buffer_size
        self.index  = 0
        self.length = 0

        self.obs         = np.zeros((buffer_size, num_obs),  dtype=np.float32)
        self.next_obs    = np.zeros((buffer_size, num_obs),  dtype=np.float32)
        self.act         = np.zeros((buffer_size, num_act),  dtype=np.float32)
        self.rew         = np.zeros(buffer_size,             dtype=np.float32)
        self.done        = np.zeros(buffer_size,             dtype=bool)
        self.dhat        = np.zeros((buffer_size, dob_dim),  dtype=np.float32)
        self.dx_nom      = np.zeros((buffer_size, num_obs),  dtype=np.float32)
        self.uncertainty = np.zeros((buffer_size, dob_dim),  dtype=np.float32)

    def store(self, obs, act, next_obs, rew, done, dhat, dx_nom, uncertainty):
        idx = self.index
        self.obs[idx]         = obs
        self.act[idx]         = act
        self.next_obs[idx]    = next_obs
        self.rew[idx]         = rew
        self.done[idx]        = done
        self.dhat[idx]        = dhat
        self.dx_nom[idx]      = dx_nom
        self.uncertainty[idx] = uncertainty
        self.index  = (self.index + 1) % self.size
        self.length = min(self.length + 1, self.size)

    def store_batch(self, obs_b, act_b, next_obs_b, rew_b, done_b):
        """Batch store with wrap-around (model buffer 용도)."""
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
