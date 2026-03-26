import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from scipy.stats import uniform, randint

import data as dt  # re-use your existing feature engineering + split_data

try:
    import xgboost as xgb
except ImportError as e:
    raise ImportError("xgboost is required. Install with: pip install xgboost") from e


class Arguments:
    """Lightweight args container (matches how the notebook builds args)."""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


# ---- Data loading: COPIED from your train.py (unchanged) ----
def load_dataset(args):
    """Load and prepare all datasets for leakage-safe forecasting."""
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "Data")
    
    btr_data = pd.read_csv(os.path.join(data_dir, "cordata.csv"))
    btr_data = btr_data.set_index("Date")

    macro_data = pd.read_csv(os.path.join(data_dir, "disaggregated.csv")).fillna(0)
    macro_data = macro_data.rename(columns={'Unnamed: 3': 'Date'}).set_index('Date')

    dummy = pd.read_csv(os.path.join(data_dir, "dummy.csv")).fillna(0)
    dummy = dummy.set_index("Date")

    btr_data.index = pd.to_datetime(btr_data.index)
    macro_data.index = pd.to_datetime(macro_data.index)
    dummy.index = pd.to_datetime(dummy.index)
    
    btr_data = btr_data.sort_index()
    macro_data = macro_data.sort_index()
    dummy = dummy.sort_index()

    start = pd.to_datetime(getattr(args, 'start_date', "1992-01-01"))
    btr_data = btr_data[btr_data.index >= start]
    macro_data = macro_data[macro_data.index >= start]
    dummy = dummy[dummy.index >= start]

    df = btr_data.join(macro_data, how="inner").join(dummy, how="inner")

    feature_cols = getattr(
        args, 'features',
        ['BIR', 'BOC', 'Other Offices', 'Non-tax Revenues', 'Expenditures',
         'TotalTrade_PHPMN', 'NominalGDP_disagg', 'Pop_disagg']
    )
    label_cols = getattr(
        args, 'labels',
        ['BIR', 'BOC', 'Other Offices', 'Non-tax Revenues', 'Expenditures']
    )
    dummy_vars = getattr(args, 'dummy_vars', ['COVID-19', 'TRAIN', 'CREATE', 'FIST', 'BIR_COMM'])

    use_lags = getattr(args, 'use_lags', True)

    log_transform = getattr(args, 'log_transform', False)
    skip_log_cols = getattr(args, 'skip_log_cols', ['Inflation', 'USDPHP'])
    
    if log_transform:
        cols_to_log = set(label_cols + feature_cols) - set(skip_log_cols)
        for col in cols_to_log:
            if col in df.columns:
                df[col] = np.log1p(df[col])

    df = dt.add_seasonal_features(df)

    # Explicit target lags requested in config, e.g. lag_1, lag_3, lag_12
    if use_lags:
        df = dt.add_lag_features(df, label_cols, args.lag_periods)

    seasonal_cols = ['month_sin', 'month_cos', 'quarter_sin', 'quarter_cos',
                     'is_tax_season', 'is_year_end']
    use_seasonal = getattr(args, 'use_seasonal', True)

    feature_blocks = []
    seen_cols = set()

    # 1) Raw feature columns:
    #    - if already a lagged column name, keep as-is
    #    - otherwise, lag once
    for col in feature_cols:
        if col not in df.columns:
            print(f"Warning: Feature '{col}' not found in data")
            continue

        if "_lag_" in col:
            new_name = col
            block = df[[col]].copy()
        else:
            new_name = f"{col}_lag_1"
            block = df[[col]].shift(1).rename(columns={col: new_name})

        if new_name not in seen_cols:
            feature_blocks.append(block)
            seen_cols.add(new_name)

    # 2) Explicit target lags from args.lag_periods:
    #    keep them exactly as created, do not shift again
    if use_lags:
        explicit_lag_cols = [col for col in df.columns if '_lag_' in col]
        for col in explicit_lag_cols:
            if col not in seen_cols:
                feature_blocks.append(df[[col]])
                seen_cols.add(col)

    # 3) Dummy variables:
    #    keep contemporaneous, do NOT lag
    if dummy_vars:
        for col in dummy_vars:
            if col in df.columns and col not in seen_cols:
                feature_blocks.append(df[[col]].copy())
                seen_cols.add(col)

    # 4) Seasonal variables:
    #    keep contemporaneous, do NOT lag
    if use_seasonal:
        for col in seasonal_cols:
            if col in df.columns and col not in seen_cols:
                feature_blocks.append(df[[col]].copy())
                seen_cols.add(col)

    features_df = pd.concat(feature_blocks, axis=1)

    # Drop rows made invalid by lagging and align labels
    valid_idx = features_df.dropna().index
    features_df = features_df.loc[valid_idx]
    labels_df = df.loc[valid_idx, label_cols]

    X = features_df.values.copy()
    y = labels_df.values.copy()

    train_data, val_data, test_data = dt.split_data(X)
    train_labels, val_labels, test_labels = dt.split_data(y)

    cv_data = np.concatenate([train_data, val_data], axis=0)
    cv_labels = np.concatenate([train_labels, val_labels], axis=0)

    if getattr(args, 'return_df', False):
        return {'df': features_df, 'labels_df': labels_df}

    return {
        'cv_data': cv_data,
        'cv_labels': cv_labels,
        'test_data': test_data,
        'test_labels': test_labels,
        'input_size': cv_data.shape[1],
        'output_size': cv_labels.shape[1],
        'log_transform': log_transform,
    }

# ---- Metrics ----
def mape(y_true, y_pred, eps: float = 1e-6) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)

def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return float(np.mean(np.abs(y_true - y_pred)))

def mse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return float(np.mean((y_true - y_pred) ** 2))


def _label_slug_from_args(args) -> str:
    labels = getattr(args, "labels", None) or ["label"]
    label = labels[0] if isinstance(labels, (list, tuple)) and len(labels) else str(labels)
    return str(label).strip().lower().replace(" ", "_").replace("/", "_").replace("__", "_")


def fit_predict_xgb(args: Arguments, X_train, y_train, X_val, y_val, X_test):
    """Train via xgboost.train for maximum compatibility across xgboost versions."""
    y_train = np.asarray(y_train).reshape(-1)
    y_val   = np.asarray(y_val).reshape(-1)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval   = xgb.DMatrix(X_val, label=y_val)
    dtest  = xgb.DMatrix(X_test)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": getattr(args, "eval_metric", "mae"),
        "max_depth": int(getattr(args, "max_depth", 6)),
        "eta": float(getattr(args, "learning_rate", 0.03)),
        "subsample": float(getattr(args, "subsample", 0.8)),
        "colsample_bytree": float(getattr(args, "colsample_bytree", 0.8)),
        "min_child_weight": float(getattr(args, "min_child_weight", 1.0)),
        "alpha": float(getattr(args, "reg_alpha", 0.0)),
        "lambda": float(getattr(args, "reg_lambda", 1.0)),
        "gamma": float(getattr(args, "gamma", 0.0)),
        "seed": int(getattr(args, "seed", 42)),
    }

    tree_method = getattr(args, "tree_method", None)
    if tree_method:
        params["tree_method"] = tree_method

    num_boost_round = int(getattr(args, "n_estimators", 5000))
    early_stopping_rounds = int(getattr(args, "early_stopping_rounds", 100))

    evals_result = {}

    booster = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=num_boost_round,
    evals=[(dtrain, "train"), (dval, "val")],
    evals_result=evals_result,
    early_stopping_rounds=early_stopping_rounds if early_stopping_rounds > 0 else None,
    verbose_eval=False,
    )

    best_it = getattr(booster, "best_iteration", None)
    if best_it is not None:
        train_pred = booster.predict(dtrain, iteration_range=(0, best_it + 1))
        val_pred   = booster.predict(dval,   iteration_range=(0, best_it + 1))
        test_pred  = booster.predict(dtest,  iteration_range=(0, best_it + 1))
    else:
        train_pred = booster.predict(dtrain)
        val_pred   = booster.predict(dval)
        test_pred  = booster.predict(dtest)

    return booster, train_pred, val_pred, test_pred, evals_result

def tune_xgb_model(X_train, y_train, n_iter=100, n_splits=5):
    """
    Perform hyperparameter tuning for XGBoost using RandomizedSearchCV
    with TimeSeriesSplit to avoid temporal leakage.
    """

    param_dist = {
        "max_depth": randint(3, 12),
        "min_child_weight": randint(1, 10),
        "learning_rate": uniform(0.01, 0.49),
        "n_estimators": randint(100, 800),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
        "gamma": uniform(0, 5),
        "reg_alpha": uniform(0, 1),
        "reg_lambda": uniform(0, 5),
    }

    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        random_state=42
    )

    tscv = TimeSeriesSplit(n_splits=n_splits)

    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=tscv,
        scoring="neg_mean_squared_error",
        verbose=1,
        n_jobs=-1,
        random_state=42
    )

    random_search.fit(X_train, y_train)

    best_model = random_search.best_estimator_
    best_params = random_search.best_params_

    print("Best Hyperparameters:", best_params)

    return best_model, best_params

def run(args, dataset, save_dir=None):
    """
    Clean workflow:
    1) Tune hyperparameters using RandomizedSearchCV(cv=5)
    2) Do one final chronological train/val split on cv_data
    3) Train one final XGBoost model with best params
    4) Save GRU-style artifacts and metrics
    """
    if save_dir is None:
        save_dir = os.path.join("results", "xgb_model")
    os.makedirs(save_dir, exist_ok=True)

    # -----------------------------
    # Load dataset
    # -----------------------------
    cv_data = dataset["cv_data"]
    cv_labels = dataset["cv_labels"]
    test_data = dataset["test_data"]
    test_labels = dataset["test_labels"]

    cv_y = cv_labels[:, 0] if cv_labels.ndim == 2 else cv_labels
    test_y = test_labels[:, 0] if test_labels.ndim == 2 else test_labels

    # -----------------------------
    # Hyperparameter tuning
    # -----------------------------
    best_model, best_params = tune_xgb_model(cv_data, cv_y)

    # Put tuned params back into args so fit_predict_xgb uses them
    for k, v in best_params.items():
        setattr(args, k, v)

    # -----------------------------
    # Existing metrics check
    # -----------------------------
    metrics_path = os.path.join(save_dir, "metrics.json")
    existing_metrics = None
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            existing_metrics = json.load(f)

    # -----------------------------
    # Final chronological split
    # -----------------------------
    train_size = int(0.75 * len(cv_data))

    X_train = cv_data[:train_size]
    X_val   = cv_data[train_size:]
    
    y_train = cv_y[:train_size]
    y_val   = cv_y[train_size:]
    
    X_test = test_data
    y_test = test_y

    # -----------------------------
    # Final fit
    # -----------------------------
    booster, train_pred, val_pred, test_pred, evals_result = fit_predict_xgb(
        args, X_train, y_train, X_val, y_val, X_test
    )

    metric_name = getattr(args, "eval_metric", "mae")
    train_losses = np.asarray(evals_result["train"][metric_name], dtype=float)
    val_losses = np.asarray(evals_result["val"][metric_name], dtype=float)

    # -----------------------------
    # Metrics
    # -----------------------------
    best_it = getattr(booster, "best_iteration", None)

    metrics = {
        "train_metrics": {
            "mse": mse(y_train, train_pred),
            "rmse": rmse(y_train, train_pred),
            "mae": mae(y_train, train_pred),
            "mape": mape(y_train, train_pred),
        },
        "val_metrics": {
            "mse": mse(y_val, val_pred),
            "rmse": rmse(y_val, val_pred),
            "mae": mae(y_val, val_pred),
            "mape": mape(y_val, val_pred),
        },
        "test_metrics": {
            "mse": mse(y_test, test_pred),
            "rmse": rmse(y_test, test_pred),
            "mae": mae(y_test, test_pred),
            "mape": mape(y_test, test_pred),
        },
        "best_iteration": int(best_it) if best_it is not None else -1,
        "best_params": best_params,
        "eval_metric": metric_name,
    }

    # -----------------------------
    # Improvement check
    # -----------------------------
    if existing_metrics:
        previous_best_rmse = existing_metrics.get("test_metrics", {}).get("rmse", float("inf"))
        if metrics["test_metrics"]["rmse"] >= previous_best_rmse:
            print(
                f"Performance did not improve "
                f"(RMSE: {metrics['test_metrics']['rmse']:.4f} vs {previous_best_rmse:.4f}). "
                f"Keeping previous model."
            )
            return booster, metrics

    # -----------------------------
    # Save artifacts
    # -----------------------------

    np.save(os.path.join(save_dir, "train_predictions.npy"), train_pred)
    np.save(os.path.join(save_dir, "train_actuals.npy"), y_train)
    
    np.save(os.path.join(save_dir, "val_predictions.npy"), val_pred)
    np.save(os.path.join(save_dir, "val_actuals.npy"), y_val)
    
    np.save(os.path.join(save_dir, "test_predictions.npy"), test_pred)
    np.save(os.path.join(save_dir, "test_actuals.npy"), y_test)
    
    np.save(os.path.join(save_dir, "train_losses.npy"), train_losses)
    np.save(os.path.join(save_dir, "val_losses.npy"), val_losses)
    
    booster.save_model(os.path.join(save_dir, "best_model.json"))
    
    args_dict = args.__dict__.copy()
    args_dict["return_df"] = True
    dataset_df = load_dataset(Arguments(**args_dict))
    
    with open(os.path.join(save_dir, "feature_names.json"), "w") as f:
        json.dump(list(dataset_df["df"].columns), f, indent=4)
    
    with open(os.path.join(save_dir, "best_params.json"), "w") as f:
        json.dump(best_params, f, indent=4)
    
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print("Performance improved! Saving new best model and metrics.")
    return booster, metrics