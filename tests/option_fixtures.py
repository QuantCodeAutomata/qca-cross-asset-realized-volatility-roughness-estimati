"""Entirely artificial option definitions and prices, not stored market data."""
import numpy as np
import pandas as pd
from src.option_pricing import black76_price


def price_fixture(venue="OPRA", n_strikes=21, n_dates=4, days=(14, 30, 90, 365),
                  positive_volume=True, H=.23, weak=False, roots=("SPX",), rate=.02):
    definitions, prices, context = [], [], []
    rng = np.random.default_rng(99)
    for di, date in enumerate(pd.bdate_range("2024-01-02", periods=n_dates)):
        for day in days:
            expiry = date + pd.Timedelta(days=day)
            T = day / 365
            forward = 100 + di
            slope = (-.08 * T**(H-.5) if not weak else
                     -np.exp(rng.normal(-3, 1.0)))
            context.append({"date": date, "underlying_id": "U", "expiry": expiry,
                            "futures_settlement": forward, "underlying_contract_id": "FUT_"+str(day)})
            for root in roots:
                for ki, k in enumerate(np.linspace(-.07, .07, n_strikes)):
                    strike = forward * np.exp(k)
                    iv = .25 + slope * k + .1*k*k
                    # Weak tests require valid IVs even for a steep random smile.
                    iv = max(iv, .03)
                    for kind in ("call", "put"):
                        cid = f"{root}_{day}_{ki}_{kind}"
                        definitions.append({"date": date, "contract_id": cid, "underlying_id": "U",
                            "root": root, "venue": venue, "expiry": expiry, "strike": strike,
                            "option_type": kind, "exercise_style": "european", "underlying_contract_id": "FUT_"+str(day)})
                        prices.append({"date": date, "contract_id": cid,
                            "price": black76_price(forward, strike, T, rate, iv, kind),
                            "volume": (20 if root.endswith("W") else 10) if positive_volume else 0})
    return pd.DataFrame(definitions), pd.DataFrame(prices), pd.DataFrame(context)
