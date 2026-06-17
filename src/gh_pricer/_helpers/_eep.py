class _EepAdjuster:
    """Subtracts early-exercise premium from American puts for dte > 0."""
    def __init__(self, rate: float):
        self._rate = rate

    def __call__(self, puts, strikes, dte: int, weekday):
        if dte == 0:
            return puts
        if weekday is not None:
            expiry_wd     = (weekday + dte) % 5
            full_wks, ext = divmod(dte, 5)
            wknd_cross    = 1 if (weekday + ext) >= 5 else 0
            cal_days      = full_wks * 7 + ext + wknd_cross * 2
            settl_off     = (3 if expiry_wd == 4 else 1) - (3 if weekday == 4 else 1)
            interest_days = cal_days + settl_off
        else:
            interest_days = dte
        return puts - self._rate * interest_days / 365 * strikes


class _NoEep:
    def __call__(self, puts, *_):
        return puts
