import gym
import numpy as np
import pandas as pd

# ---------- helper ----------------------------------------------------------- #
def zscore(series: pd.Series, window: int):
    roll_mean = series.rolling(window, min_periods=1).mean()
    roll_std  = (
        series.rolling(window, min_periods=1).std()
              .fillna(1e-8)                # ① replace NaN stdev
              .replace(0, 1e-8)            # ② just in case
    )
    return (series - roll_mean) / roll_std

def zone_from_z(z, ot, ct):
    """Map z-score to the 5 zones used in the paper."""
    if   z >  ot: return  2         # Short  zone
    elif z >  ct: return  1         # Neutral-Short
    elif z < -ot: return -2         # Long   zone
    elif z < -ct: return -1         # Neutral-Long
    else:          return  0        # Close  zone

# ---------- main environment ------------------------------------------------- #
class PairTradingEnv(gym.Env):
    """
    Observation  ⟨position, z-score, zone⟩           (shape = (3,))
    RL-1 actions {0: long-leg, 1: flat, 2: short-leg}
    RL-2 action   continuous A ∈ [-1,1]  (target position %)
    """
    metadata = {"render.modes": []}

    def __init__(
        self,
        spread: pd.Series,
        rl_mode: str = "RL1",
        window: int = 900,
        open_thr: float = 1.8,
        close_thr: float = 0.4,
        fee: float = 0.0002,               # 0.02 %
    ):
        super().__init__()

        self.spread  = pd.Series(spread).reset_index(drop=True)
        self.z       = zscore(self.spread, window)
        self.window  = window
        self.ot      = open_thr
        self.ct      = close_thr
        self.fee     = fee                 # charged on |Δposition|

        # RL variant
        self.rl_mode = rl_mode.upper()
        if self.rl_mode not in {"RL1", "RL2"}:
            raise ValueError("rl_mode must be 'RL1' or 'RL2'")

        # observation space
        self.observation_space = gym.spaces.Box(
            low  = np.array([-1.0, -np.inf, -2], dtype=np.float32),
            high = np.array([ 1.0,  np.inf,  2], dtype=np.float32),
        )

        # action space
        if self.rl_mode == "RL1":
            self.action_space = gym.spaces.Discrete(3)        # 0/1/2
        else:
            self.action_space = gym.spaces.Box(
                low  = np.array([-1.0], dtype=np.float32),
                high = np.array([ 1.0], dtype=np.float32),
            )

        # state
        self._reset_internal()

    # --------------------------------------------------------------------- #
    #  RL API
    # --------------------------------------------------------------------- #
    def reset(self):
        self._reset_internal()
        return self._get_obs()

    def step(self, action):
        # ----- translate action into target position ------------------- #
        if self.rl_mode == "RL1":                 # discrete
            target_pos = {0: 1.0, 1: 0.0, 2: -1.0}[int(action)]
        else:                                     # continuous
            target_pos = float(np.clip(action, -1.0, 1.0)[0])

        # ----- costs & reward ------------------------------------------ #
        prev_spread = self.spread.iloc[self.t]
        self._apply_trade(target_pos)             # updates self.position
        self.t += 1
        done = self.t >= len(self.spread) - 1

        pnl = -self.position * (self.spread.iloc[self.t] - prev_spread)   # mean-reversion profit
        tx_cost = self.fee * abs(self.delta_pos)
        shaping = -0.05 * abs(self.delta_pos)                            # small penalty -> fewer flips
        reward = pnl - tx_cost + shaping

        return (self._get_obs() if not done else np.zeros(3, dtype=np.float32),
                reward,
                done,
                {})

    # --------------------------------------------------------------------- #
    #  helpers
    # --------------------------------------------------------------------- #
    def _reset_internal(self):
        self.t         = 0
        self.position  = 0.0   # current portfolio percentage (−1 … +1)
        self.delta_pos = 0.0

    def _get_obs(self):
        z_t   = self.z.iloc[self.t]
        zone  = zone_from_z(z_t, self.ot, self.ct)
        obs   = np.array([self.position, z_t, zone], dtype=np.float32)
        return obs

    def _apply_trade(self, target_pos):
        """Adjusts position, stores delta for cost calculation."""
        self.delta_pos = target_pos - self.position
        self.position  = target_pos
