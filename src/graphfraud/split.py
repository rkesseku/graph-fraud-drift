"""Strict temporal split used throughout.

Defaults follow the common Elliptic protocol (Weber et al., 2019): train on time steps 1-34 and
test on 35-49. Inside the training window, steps 30-34 are held out as a *validation* block for
early stopping and threshold selection; the final models are then refit on all of 1-34.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TemporalSplit:
    val_start: int = 30
    train_end: int = 34
    dark_market_step: int = 43  # the dark-market shutdown discussed in the Elliptic paper

    def masks(self, step: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
        if not 1 < self.val_start <= self.train_end:
            raise ValueError("need 1 < val_start <= train_end")
        lab = y >= 0
        test = lab & (step > self.train_end)
        return {
            "fit": lab & (step < self.val_start),
            "val": lab & (step >= self.val_start) & (step <= self.train_end),
            "train": lab & (step <= self.train_end),
            "test": test,
            "test_pre_shutdown": test & (step < self.dark_market_step),
            "test_post_shutdown": test & (step >= self.dark_market_step),
        }
