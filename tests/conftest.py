"""
Shared pytest fixtures for gh_pricer tests.

Builds minimal in-memory CalTable and CteTable without touching disk.
"""

import pytest

from gh_pricer.types import GhEntry, CalTable, CteTable


# Realistic-ish GH parameters: m≈0, h=1.0, mu_y=0, sigma_y=1.
_ENTRY = GhEntry(m=0.0, h=1.0, mu_y=0.0, sigma_y=1.0)

BAR_SIZE = 5   # minutes

# Session: 09:30..16:00, bar_size=5 → 78 bars per day
_SESSION_START_MIN = 9 * 60 + 30   # 570
_SESSION_END_MIN   = 16 * 60        # 960


def _make_cal_table() -> CalTable:
    entries = {
        mins: _ENTRY
        for mins in range(_SESSION_START_MIN, _SESSION_END_MIN, BAR_SIZE)
    }
    return CalTable(
        entries    = entries,
        expiry_bar = _SESSION_END_MIN,
        symbol     = 'TEST',
        bar_size   = BAR_SIZE,
    )


def _make_cte_table() -> CteTable:
    entries = {
        mins: _ENTRY
        for mins in range(_SESSION_START_MIN, _SESSION_END_MIN, BAR_SIZE)
    }
    return CteTable(
        entries   = entries,
        vol_scale = 1.0,
        n_neg     = 16,
        n_pos     = 16,
    )


@pytest.fixture
def cal_table():
    return _make_cal_table()


@pytest.fixture
def cte_table():
    return _make_cte_table()


@pytest.fixture
def symbol_pricer(cal_table, cte_table):
    from gh_pricer.symbol_pricer import SymbolPricer
    return SymbolPricer(
        cal_table_0dte = cal_table,
        cte_tables     = {1: cte_table, 5: cte_table},
        bar_size       = BAR_SIZE,
        rate           = 0.045,
        is_american    = True,
        min_tick       = None,     # no rounding in tests
        bars_per_day   = 78,
        rounding       = False,
        price_with_eep = False,    # keep prices clean for put-call parity checks
    )


@pytest.fixture
def open_bar() -> int:
    """First bar of the session: 09:30 → total_minutes = 570."""
    return _SESSION_START_MIN
