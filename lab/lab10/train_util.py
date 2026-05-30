from __future__ import annotations

import argparse
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch

# ------------------------------------------------
# WARNING: DO NOT CHANGE ANYTHING IN THIS FILE, 
# YOU SHOULD WRITE YOUR SOLUTION IN THE NOTEBOOK.
# ------------------------------------------------


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Seed all relevant RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    env_id:       str   = "CartPole-v1"
    solver:       str   = "dqn"           # 'reinforce' | 'dqn' | 'tabular_q'
    episodes:     int   = 500
    max_steps:    int   = 500             # hard cap per episode
    gamma:        float = 0.99
    lr:           float = 1e-3
    seed:         int   = 42
    render:       bool  = False
    eval_episodes:int   = 10
    save_every:   int   = 100             # 0 = never
    save_dir:     str   = "checkpoints"
    load:         str   = ""             # path to load checkpoint
    device:       str   = "cpu"
    # DQN-specific
    batch_size:         int = 64
    buffer_capacity:    int = 50_000
    target_update_freq: int = 2
    epsilon_start:      float = 1.0
    epsilon_end:        float = 0.05
    epsilon_decay_steps:int = 10_000


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

class EpisodeLogger:
    """Lightweight console + in-memory logger."""

    def __init__(self, log_every: int = 10) -> None:
        self.log_every = log_every
        self.history:  list[dict[str, Any]] = []

    def record(self, ep: int, ret: float, steps: int, extra: dict) -> None:
        record = {"episode": ep, "return": ret, "steps": steps, **extra}
        self.history.append(record)

        if ep % self.log_every == 0 or ep == 1:
            recent = [h["return"] for h in self.history[-20:]]
            avg    = np.mean(recent)
            print(
                f"[Ep {ep:>5}] return={ret:>7.1f}  "
                f"avg20={avg:>7.1f}  steps={steps:>4}  "
                + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                            for k, v in extra.items())
            )

    def summary(self) -> None:
        if not self.history:
            return
        returns = [h["return"] for h in self.history]
        episodes = [h["episode"] for h in self.history]
        print("\n" + "=" * 60)
        print(f"  Episodes   : {len(returns)}")
        print(f"  Mean return: {np.mean(returns):.2f}")
        print(f"  Max return : {np.max(returns):.2f}")
        print(f"  Last 50 avg: {np.mean(returns[-50:]):.2f}")
        print("=" * 60)

        plt.figure(figsize=(10, 4))
        plt.plot(episodes, returns, linewidth=0.8, alpha=0.5, label="Return")
        window = min(20, len(returns))
        smoothed = np.convolve(returns, np.ones(window) / window, mode="valid")
        plt.plot(episodes[window - 1:], smoothed, linewidth=2, label=f"Avg{window}")
        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.title("Training Return")
        plt.legend()
        plt.tight_layout()
        plt.show()