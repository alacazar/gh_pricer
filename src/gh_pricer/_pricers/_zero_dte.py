import numpy as np

from ..types import CalTable, GreeksChain

_SQRT2       = np.sqrt(2.0)
_SQRTPI      = np.sqrt(np.pi)
_INV_SQRT2PI = 1.0 / np.sqrt(2.0 * np.pi)
_X_CLIP      = 5.0
_VEGA_BUMP   = 1e-4


class ZeroDtePricer:
    """
    0DTE pricer with per-bar pre-computed quadrature nodes.

    At construction: for every GhEntry, computes and stores
        base_x = entry.m + entry.h * sinh(mu_y + sigma_y * √2 * t_nodes)
    At price time: x = clip(spot_vol * base_x, ...) — one scalar broadcast,
    no sinh.
    """

    def __init__(
        self,
        cal_table: CalTable,
        t_nodes:   np.ndarray,
        w_nodes:   np.ndarray,
        bar_size:  int,
    ):
        self._expiry_bar = cal_table.expiry_bar
        self._bar_size   = bar_size
        self._w          = w_nodes

        n = max(cal_table.entries, default=0) + 1
        self._entries: list = [None] * n   # GhEntry | None — needed for gamma
        self._base_x:  list = [None] * n   # np.ndarray | None

        for m, e in cal_table.entries.items():
            y = e.mu_y + e.sigma_y * _SQRT2 * t_nodes
            self._entries[m] = e
            self._base_x[m]  = e.m + e.h * np.sinh(y)

    # ------------------------------------------------------------------

    def price(
        self,
        spot_price: float,
        spot_vol:   float,
        strikes:    np.ndarray,
        bar:        int,
    ) -> tuple:
        if bar >= self._expiry_bar:
            return (np.maximum(spot_price - strikes, 0.0),
                    np.maximum(strikes - spot_price, 0.0))
        base_x = self._base_x[bar] if bar < len(self._base_x) else None
        if base_x is None:
            raise KeyError(f"No 0DTE GhEntry for bar {bar}.")
        return self._price(spot_price, spot_vol, strikes, base_x)

    def greeks(
        self,
        spot_price: float,
        spot_vol:   float,
        strikes:    np.ndarray,
        bar:        int,
    ) -> GreeksChain:
        if bar >= self._expiry_bar:
            zeros    = np.zeros_like(strikes)
            itm_call = (spot_price > strikes).astype(float)
            return GreeksChain(strikes, itm_call, itm_call - 1.0,
                               zeros, zeros, zeros, zeros, zeros)

        base_x = self._base_x[bar] if bar < len(self._base_x) else None
        entry  = self._entries[bar] if bar < len(self._entries) else None
        if base_x is None:
            raise KeyError(f"No 0DTE GhEntry for bar {bar}.")

        # Delta — analytical via quadrature indicator
        x     = np.clip(spot_vol * base_x, -_X_CLIP, _X_CLIP)
        exp_x = np.exp(x)
        S_T   = spot_price * exp_x
        itm   = (S_T > strikes[:, None]).astype(float)
        dc    = (itm * exp_x) @ self._w / _SQRTPI
        dp    = dc - 1.0

        # Gamma — analytical via arcsinh-normal PDF
        m, h  = entry.m * spot_vol, entry.h * spot_vol
        x_K   = np.log(strikes / spot_price)
        u     = (x_K - m) / h
        z     = (np.arcsinh(u) - entry.mu_y) / entry.sigma_y
        phi_z = np.exp(-0.5 * z * z) * _INV_SQRT2PI
        gamma = phi_z / (entry.sigma_y * spot_price * np.sqrt(h * h + (x_K - m) ** 2))

        # Vega — central difference; base_x is the same for ±dv bumps
        dv = _VEGA_BUMP
        c_up, p_up = self._price(spot_price, spot_vol + dv, strikes, base_x)
        c_dn, p_dn = self._price(spot_price, spot_vol - dv, strikes, base_x)
        inv2dv    = 0.5 / dv
        vega_call = (c_up - c_dn) * inv2dv
        vega_put  = (p_up - p_dn) * inv2dv

        # Theta — next-bar price change at current vol
        next_bar    = bar + self._bar_size
        next_base_x = self._base_x[next_bar] if next_bar < len(self._base_x) else None
        if next_base_x is not None:
            diff  = S_T - strikes[:, None]
            c_now = np.maximum( diff, 0.0) @ self._w / _SQRTPI
            p_now = np.maximum(-diff, 0.0) @ self._w / _SQRTPI
            c_nxt, p_nxt = self._price(spot_price, spot_vol, strikes, next_base_x)
            theta_call = c_nxt - c_now
            theta_put  = p_nxt - p_now
        else:
            theta_call = theta_put = np.zeros_like(strikes)

        return GreeksChain(strikes, dc, dp, gamma,
                           vega_call, vega_put, theta_call, theta_put)

    def price_and_greeks(
        self,
        spot_price: float,
        spot_vol:   float,
        strikes:    np.ndarray,
        bar:        int,
    ) -> tuple:
        """Combined 0DTE price + greeks. Shares S_T between price and delta,
        and reuses the base price as theta's c_now."""
        if bar >= self._expiry_bar:
            calls = np.maximum(spot_price - strikes, 0.0)
            puts  = np.maximum(strikes - spot_price, 0.0)
            zeros    = np.zeros_like(strikes)
            itm_call = (spot_price > strikes).astype(float)
            g = GreeksChain(strikes, itm_call, itm_call - 1.0,
                            zeros, zeros, zeros, zeros, zeros)
            return calls, puts, g

        base_x = self._base_x[bar] if bar < len(self._base_x) else None
        entry  = self._entries[bar] if bar < len(self._entries) else None
        if base_x is None:
            raise KeyError(f"No 0DTE GhEntry for bar {bar}.")

        x     = np.clip(spot_vol * base_x, -_X_CLIP, _X_CLIP)
        exp_x = np.exp(x)
        S_T   = spot_price * exp_x
        diff  = S_T - strikes[:, None]
        w     = self._w
        inv_sqrtpi = 1.0 / _SQRTPI

        # Price (shared S_T).
        calls = np.maximum( diff, 0.0) @ w * inv_sqrtpi
        puts  = np.maximum(-diff, 0.0) @ w * inv_sqrtpi

        # Delta (shared S_T).
        itm = (S_T > strikes[:, None])
        dc  = (np.where(itm, exp_x, 0.0)) @ w * inv_sqrtpi
        dp  = dc - 1.0

        # Gamma — analytical.
        m, h  = entry.m * spot_vol, entry.h * spot_vol
        x_K   = np.log(strikes / spot_price)
        u     = (x_K - m) / h
        z     = (np.arcsinh(u) - entry.mu_y) / entry.sigma_y
        phi_z = np.exp(-0.5 * z * z) * _INV_SQRT2PI
        gamma = phi_z / (entry.sigma_y * spot_price * np.sqrt(h * h + (x_K - m) ** 2))

        # Vega — central difference, shared base_x.
        dv = _VEGA_BUMP
        c_up, p_up = self._price(spot_price, spot_vol + dv, strikes, base_x)
        c_dn, p_dn = self._price(spot_price, spot_vol - dv, strikes, base_x)
        inv2dv    = 0.5 / dv
        vega_call = (c_up - c_dn) * inv2dv
        vega_put  = (p_up - p_dn) * inv2dv

        # Theta — reuse base price as c_now.
        next_bar    = bar + self._bar_size
        next_base_x = self._base_x[next_bar] if next_bar < len(self._base_x) else None
        if next_base_x is not None:
            c_nxt, p_nxt = self._price(spot_price, spot_vol, strikes, next_base_x)
            theta_call = c_nxt - calls
            theta_put  = p_nxt - puts
        else:
            theta_call = theta_put = np.zeros_like(strikes)

        g = GreeksChain(strikes, dc, dp, gamma,
                        vega_call, vega_put, theta_call, theta_put)
        return calls, puts, g

    def _price(
        self,
        spot_price: float,
        spot_vol:   float,
        strikes:    np.ndarray,
        base_x:     np.ndarray,
    ) -> tuple:
        x    = np.clip(spot_vol * base_x, -_X_CLIP, _X_CLIP)
        S_T  = spot_price * np.exp(x)
        diff = S_T - strikes[:, None]
        calls = np.maximum( diff, 0.0) @ self._w / _SQRTPI
        puts  = np.maximum(-diff, 0.0) @ self._w / _SQRTPI
        return calls, puts
