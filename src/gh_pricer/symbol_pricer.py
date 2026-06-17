"""
SymbolPricer — hot pricer for a single symbol.

All calibration tables and GH quadrature nodes are pre-loaded at construction.
Per-call overhead is two array index operations plus the quadrature itself.

Usage
-----
    pricer = SymbolPricer.from_disk(
        lib_folder  = '/path/to/lib',
        symbol      = 'SPY',
        bar_size    = 5,
        rate        = 0.045,
        is_american = True,
        min_tick    = 0.01,
        bars_per_day= 78,
    )

    prices = pricer.price(dte=1, spot_price=668.5, spot_vol=0.000300,
                          strikes=[665, 668, 670], bar=570)

    g = pricer.greeks(dte=1, spot_price=668.5, spot_vol=0.000300,
                      strikes=[665, 668, 670], bar=570)
"""

import numpy as np
from numpy.typing import ArrayLike

from .types import CalTable, Chain, GreeksChain
from .loader import CalTableLoader
from ._helpers import _PostProcessor
from ._pricers import ZeroDtePricer, NDtePricer


class SymbolPricer:
    """
    Hot pricer for a single symbol.

    Construct via SymbolPricer.from_disk() or pass tables directly.
    Delegates to ZeroDtePricer (dte=0) or a per-DTE NDtePricer.

    Parameters fixed at construction
    ---------------------------------
    rounding       : round prices to min_tick (True by default).
    price_with_eep : subtract EEP from American puts (True by default).
    annualize      : scale vega to annualized vol units, theta to trading days
                     (False by default — raw per-bar units).
    n_points       : Gauss-Hermite quadrature points for 0DTE (default 32).
    """

    def __init__(
        self,
        cal_table_0dte: CalTable,
        cte_tables:     dict,           # {dte: CteTable}
        bar_size:       int,
        rate:           float,
        is_american:    bool,
        min_tick:       float | None,
        bars_per_day:   int,
        rounding:       bool = True,
        price_with_eep: bool = True,
        annualize:      bool = False,
        n_points:       int  = 32,
    ):
        t_0dte, w_0dte = np.polynomial.hermite.hermgauss(n_points)
        self._zero = ZeroDtePricer(cal_table_0dte, t_0dte, w_0dte, bar_size)

        self._ndte: dict[int, NDtePricer] = {}
        for dte, tbl in cte_tables.items():
            self._ndte[dte] = NDtePricer(cal_table_0dte, tbl, bar_size)

        self._post = _PostProcessor.build(
            rate, is_american, price_with_eep, min_tick, rounding, bars_per_day, annualize,
        )

    @classmethod
    def from_disk(
        cls,
        lib_folder:     str,
        symbol:         str,
        bar_size:       int,
        rate:           float,
        is_american:    bool,
        min_tick:       float | None,
        bars_per_day:   int,
        rounding:       bool = True,
        price_with_eep: bool = True,
        annualize:      bool = False,
        n_points:       int  = 32,
    ) -> 'SymbolPricer':
        """Load calibration tables from disk and return a ready-to-price SymbolPricer."""
        loader = CalTableLoader(symbol, bar_size, lib_folder)
        cal_0dte = loader.get_cal_table(0)
        if cal_0dte is None:
            raise FileNotFoundError(
                f"No 0DTE cal table found for {symbol} bar_size={bar_size} in {lib_folder}"
            )
        cte_tables = {
            dte: loader.get_cte_table(dte)
            for dte in loader.available_dtes
        }
        return cls(
            cal_table_0dte = cal_0dte,
            cte_tables     = cte_tables,
            bar_size       = bar_size,
            rate           = rate,
            is_american    = is_american,
            min_tick       = min_tick,
            bars_per_day   = bars_per_day,
            rounding       = rounding,
            price_with_eep = price_with_eep,
            annualize      = annualize,
            n_points       = n_points,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def price(
        self,
        dte:        int,
        spot_price: float,
        spot_vol:   float,
        strikes:    ArrayLike,
        bar:        int,
        weekday:    int | None = None,
    ) -> Chain:
        """
        Price calls and puts at given DTE.

        bar     : total_minutes since 00:00 (e.g. 570 for 09:30).
        weekday : 0=Mon … 4=Fri — required only when EEP weekday correction is active.

        Returns Chain(strikes, calls, puts) — parallel np.ndarray fields.
        """
        strikes = np.asarray(strikes, dtype=float)
        if dte == 0:
            calls, puts = self._zero.price(spot_price, spot_vol, strikes, bar)
        else:
            calls, puts = self._get_ndte(dte).price(spot_price, spot_vol, strikes, bar)
        calls, puts = self._post.post_price(calls, puts, strikes, dte, weekday)
        return Chain(strikes=strikes, calls=calls, puts=puts)

    def greeks(
        self,
        dte:        int,
        spot_price: float,
        spot_vol:   float,
        strikes:    ArrayLike,
        bar:        int,
    ) -> GreeksChain:
        """
        Compute option greeks at given DTE.

        bar : total_minutes since 00:00.

        Returns GreeksChain with np.ndarray fields (one entry per strike):
            delta_call, delta_put  — ∂price/∂spot
            gamma                  — ∂²price/∂spot²
            vega_call, vega_put    — ∂price/∂spot_vol (numerical central diff)
            theta_call, theta_put  — price change per bar (or per day if annualize=True)
        """
        strikes = np.asarray(strikes, dtype=float)
        if dte == 0:
            result = self._zero.greeks(spot_price, spot_vol, strikes, bar)
        else:
            result = self._get_ndte(dte).greeks(spot_price, spot_vol, strikes, bar)
        return self._post.post_greeks(result)

    def price_and_greeks(
        self,
        dte:        int,
        spot_price: float,
        spot_vol:   float,
        strikes:    ArrayLike,
        bar:        int,
        weekday:    int | None = None,
    ) -> tuple:
        """
        Compute price and greeks in one pass.

        Cheaper than calling price() + greeks() separately: shares the
        S_T tensor between price and delta, and reuses the base price as
        theta's c_now.

        Returns (Chain, GreeksChain).
        """
        strikes = np.asarray(strikes, dtype=float)
        if dte == 0:
            calls, puts, g = self._zero.price_and_greeks(spot_price, spot_vol, strikes, bar)
        else:
            calls, puts, g = self._get_ndte(dte).price_and_greeks(
                spot_price, spot_vol, strikes, bar,
            )
        calls, puts = self._post.post_price(calls, puts, strikes, dte, weekday)
        g = self._post.post_greeks(g)
        return Chain(strikes=strikes, calls=calls, puts=puts), g

    # ------------------------------------------------------------------

    def _get_ndte(self, dte: int) -> NDtePricer:
        p = self._ndte.get(dte)
        if p is None:
            raise KeyError(
                f"No CTE cal table for {dte}DTE. "
                "Run the calibration workbench to generate the .cte.tsv file first."
            )
        return p
