# Minimal scaffold: agent interface and a PPO-backed implementation (stable-baselines3)
from typing import Any, Dict, Optional
import numpy as np
import gym
from stable_baselines3 import PPO

class BaseAgent:
    def predict(self, obs: np.ndarray) -> Any:
        raise NotImplementedError

    def learn(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def save(self, path: str) -> None:
        raise NotImplementedError

    def load(self, path: str) -> None:
        raise NotImplementedError

class PpoAgent(BaseAgent):
    def __init__(self, env: gym.Env, policy: str = "MlpPolicy", **kwargs):
        self.env = env
        self.model = PPO(policy, env, verbose=0, **kwargs)

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        # Accepts either a single observation or batch; return action(s)
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return action

    def learn(self, total_timesteps: int = 10000, **kwargs):
        self.model.learn(total_timesteps=total_timesteps, **kwargs)

    def save(self, path: str) -> None:
        self.model.save(path)

    def load(self, path: str) -> None:
        self.model = PPO.load(path, env=self.env)