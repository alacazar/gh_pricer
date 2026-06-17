"""
CalTableLoader — reads calibration tables from disk for one (symbol, bar_size).

File layout (written by the research package):
    {lib_folder}/{symbol}_{bar_size}/{dte}dte.cal.tsv   — 0DTE GH table
    {lib_folder}/{symbol}_{bar_size}/{dte}dte.cte.tsv   — CTE GH table

All tables are loaded eagerly on construction and kept in memory.
"""

import os

from .types import GhEntry, CalTable, CteTable


class CalTableLoader:
    """
    Loads and caches all .cal.tsv and .cte.tsv files for one (symbol, bar_size).

    Instantiate once at startup; pass the resulting tables to SymbolPricer
    (or use SymbolPricer.from_disk which does both steps).
    """

    def __init__(self, symbol: str, bar_size: int, lib_folder: str):
        self.symbol   = symbol
        self.bar_size = bar_size
        self.folder   = os.path.join(lib_folder, f"{symbol}_{bar_size}")
        self._cal:    dict = {}   # {dte: CalTable}
        self._cte:    dict = {}   # {dte: CteTable}
        self._load()

    # ------------------------------------------------------------------ public

    def get_cal_table(self, dte: int) -> CalTable | None:
        return self._cal.get(dte)

    def get_cte_table(self, dte: int) -> CteTable | None:
        return self._cte.get(dte)

    @property
    def available_dtes(self) -> list[int]:
        """DTE values for which CTE tables are loaded."""
        return sorted(self._cte)

    # ----------------------------------------------------------------- loading

    def _load(self) -> None:
        if not os.path.isdir(self.folder):
            raise FileNotFoundError(
                f"Library folder not found: {self.folder}"
            )
        for fname in sorted(os.listdir(self.folder)):
            if fname.endswith('dte.cal.tsv'):
                dte = int(fname.split('dte')[0])
                self._cal[dte] = self._load_cal(os.path.join(self.folder, fname))
            elif fname.endswith('dte.cte.tsv'):
                dte = int(fname.split('dte')[0])
                self._cte[dte] = self._load_cte(os.path.join(self.folder, fname))

    def _load_cal(self, path: str) -> CalTable:
        entries    = {}
        symbol     = self.symbol
        bar_size   = self.bar_size
        expiry_bar = None

        col = {}
        with open(path) as f:
            for line in f:
                line = line.rstrip('\n')
                if line.startswith('#'):
                    for part in line[1:].strip().split('\t'):
                        k, v = part.split('=', 1)
                        k = k.strip()
                        if k == 'symbol':
                            symbol = v
                        elif k == 'bar_size':
                            bar_size = int(v)
                        elif k == 'expiry_bar':
                            expiry_bar = int(v)
                    continue
                if line.startswith('minutes'):
                    col = {name: i for i, name in enumerate(line.split('\t'))}
                    continue
                if not col:
                    continue
                parts = line.split('\t')
                mins = int(parts[col['minutes']])
                entries[mins] = GhEntry(
                    m       = float(parts[col['m_star']]),
                    h       = float(parts[col['h_star']]),
                    mu_y    = float(parts[col['mu_y_star']]),
                    sigma_y = float(parts[col['sigma_y_star']]),
                )

        if expiry_bar is None:
            raise ValueError(f"Missing expiry_bar in {path}")

        return CalTable(
            entries    = entries,
            expiry_bar = expiry_bar,
            symbol     = symbol,
            bar_size   = bar_size,
        )

    def _load_cte(self, path: str) -> CteTable:
        entries       = {}
        vol_scale     = 1.0
        n_points      = None
        n_neg         = None
        n_pos         = None

        col = {}
        with open(path) as f:
            for line in f:
                line = line.rstrip('\n')
                if line.startswith('#'):
                    for part in line[1:].strip().split('\t'):
                        k, v = part.split('=', 1)
                        k = k.strip()
                        if k == 'cte_vol_scale':
                            vol_scale = float(v)
                        elif k == 'n_points':
                            n_points = int(v)
                        elif k == 'n_neg':
                            n_neg = int(v)
                        elif k == 'n_pos':
                            n_pos = int(v)
                    continue
                if line.startswith('minutes'):
                    col = {name: i for i, name in enumerate(line.split('\t'))}
                    continue
                if not col:
                    continue
                parts = line.split('\t')
                mins = int(parts[col['minutes']])
                entries[mins] = GhEntry(
                    m       = float(parts[col['m']]),
                    h       = float(parts[col['h']]),
                    mu_y    = float(parts[col['mu_y']]),
                    sigma_y = float(parts[col['sigma_y']]),
                )

        # Legacy files stored n_points instead of n_neg/n_pos.
        if n_neg is None or n_pos is None:
            n = n_points if n_points is not None else 32
            n_neg = n // 2
            n_pos = n // 2

        return CteTable(
            entries   = entries,
            vol_scale = vol_scale,
            n_neg     = n_neg,
            n_pos     = n_pos,
        )
