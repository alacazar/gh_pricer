"""
Core data types for gh_pricer.

GhEntry      — four GH-distribution parameters shared by 0DTE and CTE fits.
CalTable     — 0DTE per-bar calibration table.
CteTable     — close-to-expiry per-bar calibration table for one DTE.
Chain        — price() result: parallel arrays of calls and puts.
GreeksChain  — greeks() result: parallel arrays of all greeks.

"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GhEntry:
    """
    Four parameters of one arcsinh-normal (GH) distribution fit.

    Used for both 0DTE intraday entries and CTE entries.  At pricing time,
    m and h are scaled by spot_vol before the quadrature is evaluated.
    """
    m:       float
    h:       float
    mu_y:    float
    sigma_y: float


@dataclass
class CalTable:
    """
    0DTE per-bar GH calibration table for one symbol.

    entries     : total_minutes → GhEntry, covering every bar from session
                  open up to (but not including) expiry_bar.
    expiry_bar  : bar at which the option expires (distribution collapses to intrinsic).
    symbol      : ticker, stored as metadata.
    bar_size    : bar duration in minutes.
    """
    entries:    dict        # {total_minutes: GhEntry}
    expiry_bar: int         # total_minutes
    symbol:     str
    bar_size:   int

    def get(self, bar: int) -> GhEntry | None:
        return self.entries.get(bar)


@dataclass
class CteTable:
    """
    Close-to-expiry per-bar GH calibration table for one DTE.

    entries   : total_minutes → GhEntry  (session open … session close).
    vol_scale : multiply spot_vol by this before scaling m and h.
    n_neg     : negative-side GH nodes used (saved at calibration time).
    n_pos     : positive-side GH nodes used.
    """
    entries:   dict         # {total_minutes: GhEntry}
    vol_scale: float
    n_neg:     int
    n_pos:     int

    def get(self, total_minutes: int) -> GhEntry | None:
        return self.entries.get(total_minutes)


@dataclass
class Chain:
    """Call and put prices across a set of strikes (price() result)."""
    strikes: np.ndarray   # shape (n,)
    calls:   np.ndarray   # shape (n,)
    puts:    np.ndarray   # shape (n,)


@dataclass
class GreeksChain:
    """Option greeks across a set of strikes (greeks() result)."""
    strikes:    np.ndarray   # shape (n,)
    delta_call: np.ndarray
    delta_put:  np.ndarray
    gamma:      np.ndarray
    vega_call:  np.ndarray
    vega_put:   np.ndarray
    theta_call: np.ndarray
    theta_put:  np.ndarray
