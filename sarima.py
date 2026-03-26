import os
import itertools
import warnings
import json
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings('ignore')
# ── Metrics ────────────────────────────────────────────────
def compute_metrics(actual, pred):
    actual, pred = np.array(actual), np.array(pred)
    mape = np.mean(np.abs((actual - pred) / (actual + 1e-8))) * 100
    rmse = np.sqrt(np.mean((actual - pred) ** 2))
    return {'mape': mape, 'rmse': rmse}


# ── Grid Search ────────────────────────────────────────────
def gridsearch(train, val, p_range, d_range, q_range,
               P_range=(0,), D_range=(0,), Q_range=(0,), s_range=(0,),
               top_n=5, n_jobs=-1):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from joblib import Parallel, delayed

    train = np.asarray(train, dtype=float)
    val   = np.asarray(val, dtype=float)

    orders = list(itertools.product(p_range, d_range, q_range))
    seasonal_orders = list(itertools.product(P_range, D_range, Q_range, s_range))
    combos = list(itertools.product(orders, seasonal_orders))

    print(f"  Fitting {len(combos)} models (n_jobs={n_jobs})...")

    def _fit_one(order, seasonal_order):
        try:
            mod = SARIMAX(
                train,
                order=order,
                seasonal_order=seasonal_order,
                trend="n",
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            res = mod.fit(disp=False)

            preds = []
            res_ext = res

            for y in val:
                yhat = float(res_ext.forecast(steps=1)[0])
                preds.append(yhat)
                res_ext = res_ext.append([y], refit=False)

            preds = np.asarray(preds, dtype=float)
            mse = np.mean((val - preds) ** 2)

            return {
                'order': order,
                'seasonal_order': seasonal_order,
                'MSE': float(mse)
            }
        except Exception:
            return None

    results = Parallel(n_jobs=n_jobs)(
        delayed(_fit_one)(order, seasonal_order)
        for order, seasonal_order in combos
    )

    results = [r for r in results if r is not None]
    df = pd.DataFrame(results)
    if df.empty:
        raise ValueError("No models converged.")

    return df.sort_values("MSE").head(top_n)


# # ── Final Fit ──────────────────────────────────────────────

def fit_sarima(train, test, order, seasonal_order=(0, 0, 0, 0), walk_forward=False):
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    train = np.asarray(train, dtype=float)
    test  = np.asarray(test,  dtype=float)

    res = SARIMAX(
        train,
        order=tuple(order),
        seasonal_order=tuple(seasonal_order),
        trend="n",
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)

    if walk_forward:
        preds = []
        res_ext = res
        for y in test:
            preds.append(float(res_ext.forecast(steps=1)[0]))
            res_ext = res_ext.append([y], refit=False)  # update with observed y
        preds = np.asarray(preds, dtype=float)
    else:
        preds = np.asarray(res.forecast(steps=len(test)), dtype=float)

    metrics = compute_metrics(test, preds)
    return preds, metrics, res