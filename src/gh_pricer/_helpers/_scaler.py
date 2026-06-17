import numpy as np

from ..types import GreeksChain


class _AnnualizeScaler:
    def __init__(self, bars_per_day: int):
        self._af  = np.sqrt(252.0 * bars_per_day)
        self._bpd = bars_per_day

    def __call__(self, result: GreeksChain) -> GreeksChain:
        af = self._af
        result.vega_call  = result.vega_call  / af
        result.vega_put   = result.vega_put   / af
        result.theta_call = result.theta_call * self._bpd
        result.theta_put  = result.theta_put  * self._bpd
        return result


class _NoScaler:
    def __call__(self, result: GreeksChain) -> GreeksChain:
        return result
