import gym
from gym import spaces
import numpy as np
import pandas as pd

class TradingEnvironment(gym.Env):
    def __init__(self, df, initial_balance=10000000, commission=0.0005, slippage=0.0001):
        super(TradingEnvironment, self).__init__()
        
        self.df = df
        self.initial_balance = initial_balance
        self.current_step = 0
        
        # Action space: 0 (HOLD), 1 (BUY), 2 (SELL)
        self.action_space = spaces.Discrete(3)
        
        # Observation space: [ma_diff1, ma_diff2, ma_diff3, ma_diff240, ma_diff120_240,
        #                    current_price, balance, position]
        self.observation_space = spaces.Box(
            low=np.array([-np.inf] * 8),
            high=np.array([np.inf] * 8),
            dtype=np.float32
        )
        
        self.balance = initial_balance
        self.position = 0
        self.commission = commission  # 수수료
        self.slippage = slippage      # 슬리피지
        self.done = False

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.done = False
        return self._get_observation()
    
    def _get_observation(self):
        if self.current_step < len(self.df):
            return np.array([
                self.df.iloc[self.current_step]['open'],
                self.df.iloc[self.current_step]['high'],
                self.df.iloc[self.current_step]['low'],
                self.df.iloc[self.current_step]['close'],
                self.df.iloc[self.current_step]['volume'],
                self.initial_balance,
                self.balance,
                self.position
            ], dtype=np.float32)
        else:
            return np.array([0] * 8, dtype=np.float32)  # Return a default observation
    
    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, {}

        reward = 0
        if self.current_step < len(self.df) - 1:
            current_price = self.df.iloc[self.current_step]['close']
            next_price = self.df.iloc[self.current_step + 1]['close']

            # Buy
            if action == 1 and self.balance > 0:
                self.position = self.balance / (current_price * (1 + self.slippage))
                self.balance = 0
                reward = 0  # Reward is calculated on sell
            # Sell
            elif action == 2 and self.position > 0:
                self.balance = self.position * (next_price * (1 - self.slippage))
                self.position = 0
                reward = (self.balance - self.initial_balance) / self.initial_balance
            # Hold
            else:
                reward = 0

            self.current_step += 1
        else:
            self.done = True
            reward = 0

        observation = self._get_observation()
        if self.current_step >= len(self.df) - 1:
            self.done = True

        return observation, reward, self.done, {}