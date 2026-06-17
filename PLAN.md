# gh_pricer — production library plan

## Goal

A minimal, pip-installable options pricing library with no calibration code, no path
libraries, and no config-file dependencies.  The caller pre-loads everything once at
startup and keeps it in memory; per-call overhead is a handful of dict lookups plus
the quadrature computation.

## Project layout

```
gh_pricer/
├── src/
│   └── gh_pricer/
│       ├── __init__.py          public API surface
│       ├── bar_time.py          BarTime — intraday bar identifier
│       ├── types.py             GhEntry, CalTable, CteTable
│       ├── loader.py            CalTableLoader — reads .cal.tsv / .cte.tsv from disk
│       ├── symbol_pricer.py     SymbolPricer — hot pricer, all tables in memory
│       └── _quadrature/
│           ├── __init__.py
│           ├── gh_pricer.py     price_gh, greeks_gh
│           └── gh_compound_pricer.py  price_gh_compound, greeks_gh_compound, _select_gh_nodes
├── tests/
│   ├── conftest.py
│   └── test_symbol_pricer.py
├── pyproject.toml
└── README.md
```

## Type consolidation

The research package had three separate types carrying calibration baggage:
- `CalibrationEntry`  (m_star, h_star, mu_y_star, sigma_y_star + unused extras)
- `CalibrationTable`  (wraps CalibrationEntries + ref_vol + build() classmethod)
- `ArcSinhFit`        (m, h, mu_y, sigma_y + many diagnostic fields)

All three reduce to four pricing-relevant floats.  Replaced by:

**`GhEntry(m, h, mu_y, sigma_y)`** — one shared dataclass for both 0DTE and CTE entries.
The "star" naming (m_star etc.) was a calibration artefact; dropped here.

**`CalTable`** — 0DTE table: `entries: dict[int, GhEntry]`, `expiry_bar`, `symbol`, `bar_size`.
  No `ref_vol`, no `build()`.

**`CteTable`** — CTE table: `entries: dict[int, GhEntry]`, `vol_scale`, `n_neg`, `n_pos`.
  Replaces the raw tuple `(dict, float, int, int)` stored in the research package.

## Quadrature files

Both source files contain a "core kernel" (takes raw floats/arrays, numpy only) and
a "production interface" (takes CalibrationTable + BarTime).  Only the core kernels
are needed here; `SymbolPricer` handles the table lookups.

- `price_surface_gh` / `greeks_surface_gh` — dropped (stayed in research package)
- `price_surface_gh_compound` / `greeks_surface_gh_compound` — dropped

## Factory

`SymbolPricer.from_disk(lib_folder, symbol, bar_size, rate, is_american, min_tick,
                        bars_per_day, ...)` — no Settings, no PriceEngine, no pricer.json.

## What stays in the research package

data_manager, symbol_workbench, realized_pnl_calibration, compound_fit, arcsinh_fit
(fitting), distribution_fit, path_library, pools/, reports/, market_data, PriceEngine,
Settings, RealPriceEngine.

The research package adds gh_pricer as a dependency and continues to write the same
.cal.tsv / .cte.tsv file format that CalTableLoader reads.

## Single runtime dependency

numpy only.  scipy is used by calibration code (arcsinh fitting, scipy.optimize) which
stays in the research package.
