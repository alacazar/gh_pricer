# gh-pricer

GH-quadrature options pricer for intraday and multi-day expiries.

## Overview

`gh_pricer` prices equity options under the arcsinh-normal (GH) distribution using
Gauss-Hermite quadrature. It supports:

- **0DTE** — single intraday quadrature over calibrated GH distribution
- **N-DTE** — compound integral: CTE (close-to-expiry) outer × 0DTE inner

Calibration tables are produced by a separate calibration project and loaded from disk.
All quadrature nodes are pre-computed at construction time; per-call cost is a few
array operations.

## Installation

```bash
pip install -e .
```

## Usage

```python
from gh_pricer import SymbolPricer

pricer = SymbolPricer.from_disk(
    lib_folder   = '/path/to/cal/library',
    symbol       = 'SPY',
    bar_size     = 5,
    rate         = 0.045,
    is_american  = True,
    min_tick     = 0.01,
    bars_per_day = 78,
)

# Price a chain — bar is total minutes since 00:00 (e.g. 570 = 09:30)
chain = pricer.price(dte=0, spot_price=560.0, spot_vol=0.00035,
                     strikes=[555, 558, 560, 562, 565], bar=570)
print(chain.calls, chain.puts)

# Greeks
g = pricer.greeks(dte=1, spot_price=560.0, spot_vol=0.00035,
                  strikes=[555, 560, 565], bar=570)
print(g.delta_call, g.gamma)
```

### Key parameters

| Parameter | Description |
|---|---|
| `bar_size` | Bar length in minutes (e.g. 5) |
| `rate` | Annual risk-free rate for EEP adjustment |
| `is_american` | Apply early-exercise premium to puts |
| `min_tick` | Minimum price increment for rounding |
| `bars_per_day` | Trading bars per day (e.g. 78 for 5-min bars) |
| `annualize` | Scale vega/theta to annualised units (default `False`) |

### Return types

`price()` returns a `Chain(strikes, calls, puts)`.

`greeks()` returns a `GreeksChain(strikes, delta_call, delta_put, gamma, vega_call, vega_put, theta_call, theta_put)`.

All fields are `np.ndarray`, one entry per strike.

## Requirements

- Python >= 3.11
- numpy >= 1.24

## Testing

```bash
pytest
```
