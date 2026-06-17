import numpy as np

from ..types import CalTable, CteTable, GreeksChain

_SQRT2       = np.sqrt(2.0)
_SQRTPI      = np.sqrt(np.pi)
_INV_SQRT2PI = 1.0 / np.sqrt(2.0 * np.pi)
_X_CLIP      = 5.0
_VEGA_BUMP   = 1e-4


class NDtePricer:
    """
    Compound GH pricer for one DTE, with per-bar pre-computed quadrature nodes.

    Stores base_x for both the inner (0DTE) and outer (CTE) integrals.
    Both use the same CTE t_nodes (as in price_gh_compound).
    At price time: v_nodes = (spot_vol * vol_scale) * base_x_cte,
                   x_in    = clip(spot_vol * base_x_in, ...) — no sinh.
    """

    def __init__(
        self,
        cal_table: CalTable,
        cte_table: CteTable,
        bar_size:  int,
        max_n:     int = 32,
    ):
        t_nodes, w_nodes = self._gh_nodes(cte_table.n_neg, cte_table.n_pos, max_n)
        self._bar_size  = bar_size
        self._vol_scale = cte_table.vol_scale
        self._w         = w_nodes

        cal_max = max(cal_table.entries, default=0)
        cte_max = max(cte_table.entries, default=0)
        n = max(cal_max, cte_max) + 1

        self._entries_in: list = [None] * n   # GhEntry | None — needed for gamma
        self._base_x_in:  list = [None] * n   # inner (0DTE) pre-computed nodes
        self._base_x_cte: list = [None] * n   # outer (CTE) pre-computed nodes

        for m, e in cal_table.entries.items():
            y = e.mu_y + e.sigma_y * _SQRT2 * t_nodes
            self._entries_in[m] = e
            self._base_x_in[m]  = e.m + e.h * np.sinh(y)

        for m, e in cte_table.entries.items():
            y = e.mu_y + e.sigma_y * _SQRT2 * t_nodes
            self._base_x_cte[m] = e.m + e.h * np.sinh(y)

    # ------------------------------------------------------------------

    def price(
        self,
        spot_price: float,
        spot_vol:   float,
        strikes:    np.ndarray,
        bar:        int,
    ) -> tuple:
        bxi, bxc, _ = self._resolve(bar)
        return self._price(spot_price, spot_vol, strikes, bxi, bxc)

    def greeks(
        self,
        spot_price: float,
        spot_vol:   float,
        strikes:    np.ndarray,
        bar:        int,
    ) -> GreeksChain:
        bxi, bxc, ein = self._resolve(bar)
        dc, gamma, _, _ = self._delta_gamma_and_price(
            spot_price, spot_vol, strikes, bxi, bxc, ein, want_price=False,
        )
        dp = dc - 1.0

        dv = _VEGA_BUMP
        c_up, p_up = self._price(spot_price, spot_vol + dv, strikes, bxi, bxc)
        c_dn, p_dn = self._price(spot_price, spot_vol - dv, strikes, bxi, bxc)
        inv2dv    = 0.5 / dv
        vega_call = (c_up - c_dn) * inv2dv
        vega_put  = (p_up - p_dn) * inv2dv

        theta_call, theta_put = self._theta(spot_price, spot_vol, strikes, bxi, bxc, bar)

        return GreeksChain(strikes, dc, dp, gamma,
                           vega_call, vega_put, theta_call, theta_put)

    def price_and_greeks(
        self,
        spot_price: float,
        spot_vol:   float,
        strikes:    np.ndarray,
        bar:        int,
    ) -> tuple:
        """Compound function — shares S_T between price and delta, and reuses
        the base price as theta's c_now (saving one full quadrature vs. calling
        price() and greeks() separately)."""
        bxi, bxc, ein = self._resolve(bar)
        dc, gamma, calls, puts = self._delta_gamma_and_price(
            spot_price, spot_vol, strikes, bxi, bxc, ein, want_price=True,
        )
        dp = dc - 1.0

        dv = _VEGA_BUMP
        c_up, p_up = self._price(spot_price, spot_vol + dv, strikes, bxi, bxc)
        c_dn, p_dn = self._price(spot_price, spot_vol - dv, strikes, bxi, bxc)
        inv2dv    = 0.5 / dv
        vega_call = (c_up - c_dn) * inv2dv
        vega_put  = (p_up - p_dn) * inv2dv

        next_bar = bar + self._bar_size
        nbxi = self._base_x_in[next_bar]  if next_bar < len(self._base_x_in)  else None
        nbxc = self._base_x_cte[next_bar] if next_bar < len(self._base_x_cte) else None
        if nbxi is not None and nbxc is not None:
            c_nxt, p_nxt = self._price(spot_price, spot_vol, strikes, nbxi, nbxc)
            theta_call = c_nxt - calls
            theta_put  = p_nxt - puts
        else:
            theta_call = theta_put = np.zeros_like(strikes)

        return calls, puts, GreeksChain(
            strikes, dc, dp, gamma, vega_call, vega_put, theta_call, theta_put,
        )

    # ------------------------------------------------------------------

    def _delta_gamma_and_price(
        self,
        spot_price: float,
        spot_vol:   float,
        strikes:    np.ndarray,
        bxi:        np.ndarray,
        bxc:        np.ndarray,
        ein,
        want_price: bool,
    ) -> tuple:
        """Vectorized core: builds the shared (n_out, n_in) S_T tensor and
        contracts it for delta, analytical gamma, and (optionally) price."""
        v_nodes = (spot_vol * self._vol_scale) * bxc
        x_in    = np.clip(spot_vol * bxi, -_X_CLIP, _X_CLIP)
        exp_v   = np.exp(v_nodes)
        exp_x   = np.exp(x_in)
        w       = self._w
        inv_pi  = 1.0 / np.pi

        # S_T[o, i] = spot * exp_v[o] * exp_x[i]
        S_T = spot_price * np.outer(exp_v, exp_x)            # (n_out, n_in)
        K   = strikes[:, None, None]                          # (n_str, 1, 1)
        itm = S_T[None, :, :] > K                             # bool (n_str, n_out, n_in)

        # Delta: sum_o sum_i  w[o] * exp_v[o] * w[i] * exp_x[i] * 1{S_T>K}
        delta_inner = np.where(itm, exp_x[None, None, :], 0.0) @ w   # (n_str, n_out)
        dc = (delta_inner @ (w * exp_v)) * inv_pi                    # (n_str,)

        # Gamma: analytical arcsinh-normal PDF, single outer quadrature.
        m_in, h_in = ein.m * spot_vol, ein.h * spot_vol
        x_K_eff = np.log(strikes / spot_price)[:, None] - v_nodes[None, :]  # (n_str, n_out)
        u       = (x_K_eff - m_in) / h_in
        z       = (np.arcsinh(u) - ein.mu_y) / ein.sigma_y
        phi_z   = np.exp(-0.5 * z * z) * _INV_SQRT2PI
        denom   = np.sqrt(h_in * h_in + (x_K_eff - m_in) ** 2)
        gamma   = ((phi_z / denom) @ (w * exp_v)) / (_SQRTPI * ein.sigma_y * spot_price)

        if want_price:
            diff  = S_T[None, :, :] - K                       # (n_str, n_out, n_in)
            calls = (np.maximum( diff, 0.0) @ w) @ w * inv_pi
            puts  = (np.maximum(-diff, 0.0) @ w) @ w * inv_pi
            return dc, gamma, calls, puts
        return dc, gamma, None, None

    def _theta(self, spot_price, spot_vol, strikes, bxi, bxc, bar) -> tuple:
        next_bar = bar + self._bar_size
        nbxi = self._base_x_in[next_bar]  if next_bar < len(self._base_x_in)  else None
        nbxc = self._base_x_cte[next_bar] if next_bar < len(self._base_x_cte) else None
        if nbxi is None or nbxc is None:
            z = np.zeros_like(strikes)
            return z, z
        c_now, p_now = self._price(spot_price, spot_vol, strikes, bxi, bxc)
        c_nxt, p_nxt = self._price(spot_price, spot_vol, strikes, nbxi, nbxc)
        return c_nxt - c_now, p_nxt - p_now

    def _price(
        self,
        spot_price:  float,
        spot_vol:    float,
        strikes:     np.ndarray,
        base_x_in:   np.ndarray,
        base_x_cte:  np.ndarray,
    ) -> tuple:
        v_nodes = (spot_vol * self._vol_scale) * base_x_cte
        x_in    = np.clip(spot_vol * base_x_in, -_X_CLIP, _X_CLIP)
        # S_T[o, i] = spot * exp(v_nodes[o]) * exp(x_in[i])
        S_T   = spot_price * np.outer(np.exp(v_nodes), np.exp(x_in))
        diff  = S_T[None, :, :] - strikes[:, None, None]         # (n_str, n_out, n_in)
        w     = self._w
        # Inner integral (contract n_in), then outer (contract n_out).
        calls = (np.maximum( diff, 0.0) @ w) @ w
        puts  = (np.maximum(-diff, 0.0) @ w) @ w
        inv_pi = 1.0 / np.pi
        return calls * inv_pi, puts * inv_pi

    @staticmethod
    def _gh_nodes(n_neg: int, n_pos: int, max_n: int) -> tuple:
        t_all, w_all = np.polynomial.hermite.hermgauss(max_n)
        n_half = max_n // 2
        t = np.concatenate([t_all[n_half - n_neg : n_half], t_all[n_half : n_half + n_pos]])
        w = np.concatenate([w_all[n_half - n_neg : n_half], w_all[n_half : n_half + n_pos]])
        return t, w

    def _resolve(self, bar: int) -> tuple:
        bxi = self._base_x_in[bar]  if bar < len(self._base_x_in)  else None
        bxc = self._base_x_cte[bar] if bar < len(self._base_x_cte) else None
        ein = self._entries_in[bar]  if bar < len(self._entries_in)  else None
        if bxi is None or ein is None:
            raise KeyError(f"No 0DTE GhEntry for bar {bar}.")
        if bxc is None:
            raise KeyError(f"No CTE GhEntry for bar {bar}.")
        return bxi, bxc, ein
