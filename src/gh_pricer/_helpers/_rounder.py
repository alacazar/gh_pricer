import numpy as np


class _Rounder:
    def __init__(self, min_tick: float):
        self._t = min_tick

    def __call__(self, calls, puts):
        t = self._t
        return (np.maximum(np.round(calls / t) * t, t),
                np.maximum(np.round(puts  / t) * t, t))


class _NoRounder:
    def __call__(self, calls, puts):
        return calls, puts
