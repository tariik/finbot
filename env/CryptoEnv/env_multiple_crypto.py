from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CryptoEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(
        self,
        config,
        lookback=1,
        initial_capital=1e6,
        buy_cost_pct=1e-3,
        sell_cost_pct=1e-3,
        gamma=0.99,
    ):
        super(CryptoEnv, self).__init__()

        self.lookback = lookback
        self.initial_total_asset = initial_capital
        self.initial_cash = initial_capital
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.max_stock = 1
        self.gamma = gamma
        self.price_array = config["price_array"]
        self.tech_array = config["tech_array"]
        self._generate_action_normalizer()
        self.crypto_num = self.price_array.shape[1]
        self.max_step = self.price_array.shape[0] - lookback - 1

        # Reset
        self.time = lookback - 1
        self.cash = self.initial_cash
        self.current_price = self.price_array[self.time]
        self.current_tech = self.tech_array[self.time]
        self.stocks = np.zeros(self.crypto_num, dtype=np.float32)

        self.total_asset = self.cash + (self.stocks * self.price_array[self.time]).sum()
        self.episode_return = 0.0
        self.gamma_return = 0.0

        """env information"""
        self.env_name = "MulticryptoEnv"
        self.state_dim = (
            1 + (self.price_array.shape[1] + self.tech_array.shape[1]) * lookback
        )
        self.action_dim = self.price_array.shape[1]

        # Define action and observation space
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.crypto_num,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )
        self.if_discrete = False
        self.target_return = 10

    def reset(self, *, seed=None, options=None) -> np.ndarray:
        super().reset(seed=seed)
        self.time = self.lookback - 1
        self.current_price = self.price_array[self.time]
        self.current_tech = self.tech_array[self.time]
        self.cash = self.initial_cash  # reset()
        self.stocks = np.zeros(self.crypto_num, dtype=np.float32)
        self.total_asset = self.cash + (self.stocks * self.price_array[self.time]).sum()

        state = self.get_state()
        return state, {}

    def step(self, actions) -> tuple[np.ndarray, float, bool, bool, dict]:
        self.time += 1

        price = self.price_array[self.time]
        for i in range(self.action_dim):
            norm_vector_i = self.action_norm_vector[i]
            actions[i] = actions[i] * norm_vector_i

        for index in np.where(actions < 0)[0]:  # sell_index:
            if price[index] > 0:  # Sell only if current asset is > 0
                sell_num_shares = min(self.stocks[index], -actions[index])
                self.stocks[index] -= sell_num_shares
                self.cash += price[index] * sell_num_shares * (1 - self.sell_cost_pct)

        for index in np.where(actions > 0)[0]:  # buy_index:
            if (
                price[index] > 0
            ):  # Buy only if the price is > 0 (no missing data in this particular date)
                buy_num_shares = min(
                    self.cash // (price[index] * (1 + self.buy_cost_pct)),
                    actions[index],
                )
                self.stocks[index] += buy_num_shares
                self.cash -= price[index] * buy_num_shares * (1 + self.buy_cost_pct)

        """update time"""
        done = self.time == self.max_step
        state = self.get_state()
        next_total_asset = self.cash + (self.stocks * self.price_array[self.time]).sum()
        reward = (next_total_asset - self.total_asset) * 2**-16
        self.total_asset = next_total_asset
        self.gamma_return = self.gamma_return * self.gamma + reward
        self.cumu_return = self.total_asset / self.initial_cash
        if done:
            reward = self.gamma_return
            self.episode_return = self.total_asset / self.initial_cash
        return state, reward, done, False, {}

    def get_state(self):
        state = np.hstack((self.cash * 2**-18, self.stocks * 2**-3))
        for i in range(self.lookback):
            tech_i = self.tech_array[self.time - i]
            normalized_tech_i = tech_i * 2**-15
            state = np.hstack((state, normalized_tech_i)).astype(np.float32)
        return state

    def render(self, mode="human"):
        print(f"Time: {self.time}, Total Asset: {self.total_asset}")

    def close(self):
        pass

    def _generate_action_normalizer(self):
        action_norm_vector = []
        price_0 = self.price_array[0]
        for price in price_0:
            x = len(str(price)) - 7
            action_norm_vector.append(1 / ((10) ** x))

        self.action_norm_vector = np.asarray(action_norm_vector)


if __name__ == "__main__":
    # Sample data
    price_array = np.array([
        [100, 200, 300],  # Day 1
        [110, 190, 310],  # Day 2
        [120, 195, 320],  # Day 3
        [115, 210, 330],  # Day 4
        [130, 220, 340],  # Day 5
    ])

    tech_array = np.array([
        [0.5, 0.1],  # Technical indicator day 1
        [0.6, 0.15],  # Technical indicator day 2
        [0.7, 0.2],  # Technical indicator day 3
        [0.65, 0.25],  # Technical indicator day 4
        [0.8, 0.3],  # Technical indicator day 5
    ])

    config = {
        "price_array": price_array,
        "tech_array": tech_array
    }

    env = CryptoEnv(config=config, lookback=1)

    state, _ = env.reset()
    print("Initial state:", state)

    actions = np.array(
        [0.5, -0.3, 0.0])  # Buy the first crypto, sell the second, do nothing with the third
    state, reward, done, _, _ = env.step(actions)
    print("State after one step:", state)
    print("Reward:", reward)
    print("Episode done:", done)
