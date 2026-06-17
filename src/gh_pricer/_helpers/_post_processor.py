from ._eep     import _EepAdjuster, _NoEep
from ._rounder import _Rounder,     _NoRounder
from ._scaler  import _AnnualizeScaler, _NoScaler


class _PostProcessor:
    """
    Unified post-processing for price() and greeks() results.

    Assembled once at SymbolPricer construction; the pricer owns one reference
    and calls post_price / post_greeks with no conditional logic.
    """

    def __init__(self, eep_adj, rounder, scaler):
        self._eep_adj = eep_adj
        self._rounder = rounder
        self._scaler  = scaler

    def post_price(self, calls, puts, strikes, dte: int, weekday) -> tuple:
        puts = self._eep_adj(puts, strikes, dte, weekday)
        return self._rounder(calls, puts)

    def post_greeks(self, result: 'GreeksChain') -> 'GreeksChain':
        return self._scaler(result)

    @classmethod
    def build(
        cls,
        rate:           float,
        is_american:    bool,
        price_with_eep: bool,
        min_tick:       'float | None',
        rounding:       bool,
        bars_per_day:   int,
        annualize:      bool,
    ) -> '_PostProcessor':
        return cls(
            _EepAdjuster(rate) if (is_american and price_with_eep) else _NoEep(),
            _Rounder(min_tick) if (rounding and min_tick) else _NoRounder(),
            _AnnualizeScaler(bars_per_day) if annualize else _NoScaler(),
        )
