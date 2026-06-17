"""
Unit tests for SymbolPricer.

Tests use in-memory CalTable/CteTable (no disk) via fixtures in conftest.py.
Checks shapes, signs, put-call parity, delta bounds, and gamma non-negativity.
"""

import numpy as np
import pytest

SPOT    = 100.0
VOL     = 0.001       # small per-bar vol so prices are well-defined
STRIKES = [95.0, 98.0, 100.0, 102.0, 105.0]


class TestPrice0DTE:
    def test_returns_all_strikes(self, symbol_pricer, open_bar):
        chain = symbol_pricer.price(0, SPOT, VOL, STRIKES, bar=open_bar)
        assert len(chain.strikes) == len(STRIKES)

    def test_prices_non_negative(self, symbol_pricer, open_bar):
        chain = symbol_pricer.price(0, SPOT, VOL, STRIKES, bar=open_bar)
        assert np.all(chain.calls >= 0.0)
        assert np.all(chain.puts  >= 0.0)

    def test_put_call_parity(self, symbol_pricer, open_bar):
        chain = symbol_pricer.price(0, SPOT, VOL, STRIKES, bar=open_bar)
        for i, k in enumerate(STRIKES):
            # C - P ≈ S - K; tolerance reflects 32-point GH quadrature forward error
            assert abs((chain.calls[i] - chain.puts[i]) - (SPOT - k)) < 1e-3, (
                f"Put-call parity failed at K={k}"
            )

    def test_call_monotone_in_strike(self, symbol_pricer, open_bar):
        chain = symbol_pricer.price(0, SPOT, VOL, STRIKES, bar=open_bar)
        assert np.all(np.diff(chain.calls) <= 0)

    def test_put_monotone_in_strike(self, symbol_pricer, open_bar):
        chain = symbol_pricer.price(0, SPOT, VOL, STRIKES, bar=open_bar)
        assert np.all(np.diff(chain.puts) >= 0)

    def test_intrinsic_lower_bound(self, symbol_pricer, open_bar):
        chain = symbol_pricer.price(0, SPOT, VOL, STRIKES, bar=open_bar)
        for i, k in enumerate(STRIKES):
            # 1e-3 tolerance: 32-point GH quadrature has small forward bias
            assert chain.calls[i] >= max(SPOT - k, 0.0) - 1e-3
            assert chain.puts[i]  >= max(k - SPOT, 0.0) - 1e-3


class TestGreeks0DTE:
    def test_shapes(self, symbol_pricer, open_bar):
        g = symbol_pricer.greeks(0, SPOT, VOL, STRIKES, bar=open_bar)
        n = len(STRIKES)
        for arr in (g.delta_call, g.delta_put, g.gamma,
                    g.vega_call, g.vega_put, g.theta_call, g.theta_put):
            assert len(arr) == n

    def test_delta_call_bounds(self, symbol_pricer, open_bar):
        g = symbol_pricer.greeks(0, SPOT, VOL, STRIKES, bar=open_bar)
        assert np.all(g.delta_call >= -1e-9)
        assert np.all(g.delta_call <= 1.0 + 1e-9)

    def test_delta_put_bounds(self, symbol_pricer, open_bar):
        g = symbol_pricer.greeks(0, SPOT, VOL, STRIKES, bar=open_bar)
        assert np.all(g.delta_put >= -1.0 - 1e-9)
        assert np.all(g.delta_put <= 1e-9)

    def test_delta_parity(self, symbol_pricer, open_bar):
        g = symbol_pricer.greeks(0, SPOT, VOL, STRIKES, bar=open_bar)
        np.testing.assert_allclose(g.delta_call - g.delta_put, 1.0, atol=1e-9)

    def test_gamma_non_negative(self, symbol_pricer, open_bar):
        g = symbol_pricer.greeks(0, SPOT, VOL, STRIKES, bar=open_bar)
        assert np.all(g.gamma >= -1e-12)

    def test_atm_vega_positive(self, symbol_pricer, open_bar):
        g = symbol_pricer.greeks(0, SPOT, VOL, [SPOT], bar=open_bar)
        assert g.vega_call[0] > 0.0
        assert g.vega_put[0]  > 0.0


class TestPriceNDTE:
    def test_returns_all_strikes(self, symbol_pricer, open_bar):
        chain = symbol_pricer.price(1, SPOT, VOL, STRIKES, bar=open_bar)
        assert len(chain.strikes) == len(STRIKES)

    def test_prices_non_negative(self, symbol_pricer, open_bar):
        chain = symbol_pricer.price(1, SPOT, VOL, STRIKES, bar=open_bar)
        assert np.all(chain.calls >= 0.0)
        assert np.all(chain.puts  >= 0.0)

    def test_put_call_parity(self, symbol_pricer, open_bar):
        chain = symbol_pricer.price(1, SPOT, VOL, STRIKES, bar=open_bar)
        for i, k in enumerate(STRIKES):
            # Compound pricer uses two nested GH integrals; forward error accumulates
            assert abs((chain.calls[i] - chain.puts[i]) - (SPOT - k)) < 2e-3, (
                f"Compound put-call parity failed at K={k}"
            )

    def test_ndte_ge_0dte_atm(self, symbol_pricer, open_bar):
        p0 = symbol_pricer.price(0, SPOT, VOL, [SPOT], bar=open_bar)
        p1 = symbol_pricer.price(1, SPOT, VOL, [SPOT], bar=open_bar)
        assert p1.calls[0] >= p0.calls[0] - 1e-8

    def test_unknown_dte_raises(self, symbol_pricer, open_bar):
        with pytest.raises(KeyError, match='99DTE'):
            symbol_pricer.price(99, SPOT, VOL, STRIKES, bar=open_bar)


class TestGreeksNDTE:
    def test_shapes(self, symbol_pricer, open_bar):
        g = symbol_pricer.greeks(1, SPOT, VOL, STRIKES, bar=open_bar)
        n = len(STRIKES)
        for arr in (g.delta_call, g.delta_put, g.gamma,
                    g.vega_call, g.vega_put, g.theta_call, g.theta_put):
            assert len(arr) == n

    def test_delta_call_bounds(self, symbol_pricer, open_bar):
        g = symbol_pricer.greeks(1, SPOT, VOL, STRIKES, bar=open_bar)
        assert np.all(g.delta_call >= -1e-9)
        assert np.all(g.delta_call <= 1.0 + 1e-9)

    def test_delta_parity(self, symbol_pricer, open_bar):
        g = symbol_pricer.greeks(1, SPOT, VOL, STRIKES, bar=open_bar)
        np.testing.assert_allclose(g.delta_call - g.delta_put, 1.0, atol=1e-9)

    def test_gamma_non_negative(self, symbol_pricer, open_bar):
        g = symbol_pricer.greeks(1, SPOT, VOL, STRIKES, bar=open_bar)
        assert np.all(g.gamma >= -1e-12)


class TestFromDisk:
    def test_missing_folder_raises(self, tmp_path):
        from gh_pricer.symbol_pricer import SymbolPricer
        with pytest.raises(FileNotFoundError):
            SymbolPricer.from_disk(
                lib_folder  = str(tmp_path / 'nonexistent'),
                symbol      = 'TEST',
                bar_size    = 5,
                rate        = 0.045,
                is_american = True,
                min_tick    = 0.01,
                bars_per_day= 78,
            )

    def test_missing_0dte_table_raises(self, tmp_path):
        from gh_pricer.symbol_pricer import SymbolPricer
        (tmp_path / 'TEST_5').mkdir()   # folder exists but has no .cal.tsv
        with pytest.raises(FileNotFoundError, match='0DTE'):
            SymbolPricer.from_disk(
                lib_folder  = str(tmp_path),
                symbol      = 'TEST',
                bar_size    = 5,
                rate        = 0.045,
                is_american = True,
                min_tick    = 0.01,
                bars_per_day= 78,
            )
