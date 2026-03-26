import matplotlib.pyplot as plt
import numpy as np
# import torch
# from data import inverse_transform
# import shap
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import os
import json
import pandas as pd
import shap
import xgboost as xgb
import train_xgb as tr

def plot_residual_diagnostics(actual, pred, label, save_dir=None):
    actual    = np.array(actual).flatten()
    pred      = np.array(pred).flatten()
    residuals = actual - pred
    max_lags  = min(24, len(residuals) // 2 - 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f'Residual Diagnostics — {label}')

    # Residuals over time
    axes[0,0].plot(residuals)
    axes[0,0].axhline(0, color='red', linestyle='--')
    axes[0,0].set_title('Residuals Over Time')

    # Distribution
    axes[0,1].hist(residuals, bins=20, density=True, alpha=0.7)
    xmin, xmax = axes[0,1].get_xlim()
    x = np.linspace(xmin, xmax, 100)
    axes[0,1].plot(x, stats.norm.pdf(x, residuals.mean(), residuals.std()), 'r')
    axes[0,1].set_title('Distribution')

    # ACF / PACF
    plot_acf(residuals,  lags=max_lags, ax=axes[1,0], title='ACF')
    plot_pacf(residuals, lags=max_lags, ax=axes[1,1], title='PACF')

    plt.tight_layout()
    if save_dir:
        plt.savefig(f"{save_dir}/residual_diagnostics.png", dpi=150)
    plt.show()


def plot_qq(actual_dict, pred_dict, label_cols, save_path=None):
    fig, axes = plt.subplots(1, len(label_cols), figsize=(5 * len(label_cols), 4))
    if len(label_cols) == 1:
        axes = [axes]

    for ax, label in zip(axes, label_cols):
        residuals = np.array(actual_dict[label]).flatten() - np.array(pred_dict[label]).flatten()
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title(f'Q-Q Plot — {label}')
        ax.get_lines()[1].set_color('red')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

def plot_training_history(train_losses, val_losses):
    plt.figure(figsize=(12, 5))
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Val Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_test_predictions(test_preds, test_actuals, label, save_dir=None):
    test_preds   = np.array(test_preds).flatten()
    test_actuals = np.array(test_actuals).flatten()

    plt.figure(figsize=(12, 6))
    plt.plot(test_actuals, label='Actual',    linewidth=2)
    plt.plot(test_preds,   label='Predicted', linewidth=2, alpha=0.7)
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.title(f'{label}: Predictions vs Actual')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_dir:
        plt.savefig(f"{save_dir}/predictions.png", dpi=150)
    plt.show()

def rebuild_xgb_splits(args):
    """
    Rebuild the exact chronological train/val/test splits used by train_xgb.run().

    train_xgb.load_dataset(return_df=True) returns the full feature DataFrame.
    train_xgb.run() effectively uses:
      - 60% train
      - 20% val
      - 20% test
    via:
      1) split_data() -> train/val/test
      2) concatenate train+val -> cv_data
      3) split cv_data at 75% -> final train/val
    """
    args_dict = args.__dict__.copy()
    args_dict["return_df"] = True
    df_args = tr.Arguments(**args_dict)

    dataset_df = tr.load_dataset(df_args)
    X_df = dataset_df["df"].copy()
    y_df = dataset_df["labels_df"].copy()

    n = len(X_df)
    train_end = int(0.6 * n)
    val_end = train_end + int(0.2 * n)

    train_df = X_df.iloc[:train_end].copy()
    val_df   = X_df.iloc[train_end:val_end].copy()
    test_df  = X_df.iloc[val_end:].copy()

    cv_df = pd.concat([train_df, val_df], axis=0)
    final_train_size = int(0.75 * len(cv_df))

    X_train = cv_df.iloc[:final_train_size].copy()
    X_val   = cv_df.iloc[final_train_size:].copy()
    X_test  = test_df.copy()

    return X_train, X_val, X_test, y_df


def get_y_test_from_full_y(y_df):
    if isinstance(y_df, pd.DataFrame):
        y_series = y_df.iloc[:, 0].copy()
    else:
        y_series = y_df.copy()

    n = len(y_series)
    train_end = int(0.6 * n)
    val_end = train_end + int(0.2 * n)

    y_test = y_series.iloc[val_end:].copy()
    return y_test


def load_xgb_experiment(label, start_date, feature_configs, xgb_params,
                        save_root="results_featselect"):
    """
    Shared loader for XGBoost interpretation workflows.
    Rebuilds the exact feature splits and loads the saved Booster.
    """
    start_slug = start_date.replace("-", "")
    label_slug = label.replace(" ", "_")
    save_dir = os.path.join(save_root, f"{label_slug}_{start_slug}")

    config = feature_configs[label]

    args = tr.Arguments(
        labels=[label],
        start_date=start_date,
        features=config["features"],
        dummy_vars=config["dummy_vars"],
        log_transform=False,
        use_seasonal=config["use_seasonal"],
        use_lags=config["use_lags"],
        lag_periods=config["lag_periods"],
    )

    for k, v in xgb_params.items():
        setattr(args, k, v)

    X_train, X_val, X_test, y_df = rebuild_xgb_splits(args)

    feature_names_path = os.path.join(save_dir, "feature_names.json")
    if os.path.exists(feature_names_path):
        with open(feature_names_path, "r") as f:
            feature_names = json.load(f)
        X_train = X_train[feature_names]
        X_val   = X_val[feature_names]
        X_test  = X_test[feature_names]

    model_path = os.path.join(save_dir, "best_model.json")
    booster = xgb.Booster()
    booster.load_model(model_path)

    y_test = get_y_test_from_full_y(y_df)

    return {
        "label": label,
        "start_date": start_date,
        "save_dir": save_dir,
        "booster": booster,
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_df": y_df,
        "y_test": y_test,
    }


# ============================================================
# SHAP ONLY
# ============================================================

def explain_xgb_model(label, start_date, feature_configs, xgb_params,
                      save_root="results_featselect", background_size=100):
    """
    Load one trained XGBoost model and compute SHAP values on its test set.
    """
    result = load_xgb_experiment(
        label=label,
        start_date=start_date,
        feature_configs=feature_configs,
        xgb_params=xgb_params,
        save_root=save_root,
    )

    booster = result["booster"]
    X_train = result["X_train"]
    X_test = result["X_test"]

    background = X_train.sample(min(background_size, len(X_train)), random_state=42)

    explainer = shap.TreeExplainer(booster, data=background)
    shap_values = explainer.shap_values(X_test)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    shap_importance_df = (
        pd.DataFrame({
            "feature": X_test.columns,
            "mean_abs_shap": mean_abs_shap
        })
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    result.update({
        "explainer": explainer,
        "shap_values": shap_values,
        "shap_importance_df": shap_importance_df,
    })

    return result


def shorten_feature_name(name, max_len=18):
    if len(name) <= max_len:
        return name
    return name[:max_len - 3] + "..."


def shorten_feature_names(columns, max_len=18):
    return [shorten_feature_name(col, max_len=max_len) for col in columns]


def plot_xgb_shap_summary(
    shap_values,
    X_test,
    title=None,
    save_path=None,
    max_display=8,
    figsize=(10, 6),
    shorten_names=False,
    name_max_len=16
):
    X_plot = X_test.copy()

    if shorten_names:
        X_plot.columns = shorten_feature_names(X_plot.columns, max_len=name_max_len)

    plt.figure(figsize=figsize)
    shap.summary_plot(
        shap_values,
        X_plot,
        max_display=max_display,
        show=False
    )

    if title:
        plt.title(title, fontsize=12)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_xgb_shap_bar(
    shap_values,
    X_test,
    title=None,
    save_path=None,
    max_display=8,
    figsize=(10, 5.5),
    shorten_names=False,
    name_max_len=16
):
    X_plot = X_test.copy()

    if shorten_names:
        X_plot.columns = shorten_feature_names(X_plot.columns, max_len=name_max_len)

    plt.figure(figsize=figsize)
    shap.summary_plot(
        shap_values,
        X_plot,
        plot_type="bar",
        max_display=max_display,
        show=False
    )

    if title:
        plt.title(title, fontsize=12)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_xgb_shap_waterfall(
    explainer,
    shap_values,
    X_test,
    sample_idx=0,
    title=None,
    save_path=None,
    max_display=8,
    figsize=(11, 7),
    shorten_names=False,
    name_max_len=18,
    show_feature_values=False
):
    feature_names = X_test.columns.tolist()

    if shorten_names:
        feature_names = shorten_feature_names(feature_names, max_len=name_max_len)

    explanation = shap.Explanation(
        values=shap_values[sample_idx],
        base_values=explainer.expected_value,
        data=X_test.iloc[sample_idx].values if show_feature_values else None,
        feature_names=feature_names
    )

    plt.figure(figsize=figsize)
    shap.plots.waterfall(
        explanation,
        max_display=max_display,
        show=False
    )

    if title:
        plt.title(title, fontsize=13)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

def plot_xgb_shap_combined(
    explainer,
    shap_values,
    X_test,
    sample_idx=0,
    title_prefix=None,
    save_path=None,
    summary_max_display=8,
    bar_max_display=8,
    waterfall_max_display=8,
    shorten_names=False,
    name_max_len=16,
    show_feature_values=False,
    panel_dpi=160,
    display_figsize=(24, 7)
):
    """
    Create summary, bar, and waterfall SHAP plots as separate images,
    then stitch them horizontally into one combined figure.

    Parameters
    ----------
    explainer : shap explainer
    shap_values : np.ndarray
    X_test : pd.DataFrame
    sample_idx : int
    title_prefix : str
    save_path : str or None
    shorten_names : bool
    show_feature_values : bool
        If False, waterfall omits raw feature values from labels.
    """
    import os
    import tempfile
    from PIL import Image

    feature_names = X_test.columns.tolist()
    X_plot = X_test.copy()

    if shorten_names:
        feature_names = shorten_feature_names(feature_names, max_len=name_max_len)
        X_plot.columns = feature_names

    tmpdir = tempfile.mkdtemp()

    # -------------------------
    # Summary plot
    # -------------------------
    summary_path = os.path.join(tmpdir, "summary.png")
    plt.figure(figsize=(8, 6))
    shap.summary_plot(
        shap_values,
        X_plot,
        max_display=summary_max_display,
        show=False
    )
    plt.title(
        f"{title_prefix} — Summary" if title_prefix else "SHAP Summary",
        fontsize=12
    )
    plt.tight_layout()
    plt.savefig(summary_path, dpi=panel_dpi, bbox_inches="tight")
    plt.close()

    # -------------------------
    # Bar plot
    # -------------------------
    bar_path = os.path.join(tmpdir, "bar.png")
    plt.figure(figsize=(8, 6))
    shap.summary_plot(
        shap_values,
        X_plot,
        plot_type="bar",
        max_display=bar_max_display,
        show=False
    )
    plt.title(
        f"{title_prefix} — Bar" if title_prefix else "SHAP Bar",
        fontsize=12
    )
    plt.tight_layout()
    plt.savefig(bar_path, dpi=panel_dpi, bbox_inches="tight")
    plt.close()

    # -------------------------
    # Waterfall plot
    # -------------------------
    waterfall_path = os.path.join(tmpdir, "waterfall.png")

    explanation = shap.Explanation(
        values=shap_values[sample_idx],
        base_values=explainer.expected_value,
        data=X_test.iloc[sample_idx].values if show_feature_values else None,
        feature_names=feature_names
    )

    plt.figure(figsize=(8, 6))
    shap.plots.waterfall(
        explanation,
        max_display=waterfall_max_display,
        show=False
    )
    plt.title(
        f"{title_prefix} — Waterfall (sample {sample_idx})" if title_prefix else f"SHAP Waterfall (sample {sample_idx})",
        fontsize=12
    )
    plt.tight_layout()
    plt.savefig(waterfall_path, dpi=panel_dpi, bbox_inches="tight")
    plt.close()

    # -------------------------
    # Stitch horizontally
    # -------------------------
    imgs = [Image.open(summary_path), Image.open(bar_path), Image.open(waterfall_path)]

    total_width = sum(img.width for img in imgs)
    max_height = max(img.height for img in imgs)

    combined = Image.new("RGB", (total_width, max_height), "white")

    x_offset = 0
    for img in imgs:
        combined.paste(img, (x_offset, 0))
        x_offset += img.width

    if save_path:
        combined.save(save_path)

    plt.figure(figsize=display_figsize)
    plt.imshow(combined)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    plt.close()
    
def run_xgb_shap_analysis(label, start_date, feature_configs, xgb_params,
                          save_root="results_featselect", sample_idx=0,
                          save_plots=True, background_size=100, save_csv=True):
    result = explain_xgb_model(
        label=label,
        start_date=start_date,
        feature_configs=feature_configs,
        xgb_params=xgb_params,
        save_root=save_root,
        background_size=background_size,
    )

    explainer = result["explainer"]
    shap_values = result["shap_values"]
    X_test = result["X_test"]
    save_dir = result["save_dir"]

    title_suffix = f"{label} - {start_date}"

    plot_xgb_shap_summary(
    shap_values,
    X_test,
    title=f"SHAP Summary - {title_suffix}",
    save_path=os.path.join(save_dir, "shap_summary.png") if save_plots else None,
    max_display=8,
    figsize=(10, 6),
    shorten_names=True,
    name_max_len=30
    )
    
    plot_xgb_shap_bar(
        shap_values,
        X_test,
        title=f"SHAP Bar - {title_suffix}",
        save_path=os.path.join(save_dir, "shap_bar.png") if save_plots else None,
        max_display=8,
        figsize=(10, 5.5),
        shorten_names=True,
        name_max_len=30
    )
    
    plot_xgb_shap_waterfall(
        explainer,
        shap_values,
        X_test,
        sample_idx=sample_idx,
        title=f"SHAP Waterfall - {title_suffix} - sample {sample_idx}",
        save_path=os.path.join(save_dir, f"shap_waterfall_{sample_idx}.png") if save_plots else None,
        max_display=8,
        figsize=(10, 6.5),
        shorten_names=True,
        name_max_len=30,
        show_feature_values=False
    )

    combined_path = os.path.join(save_dir, "shap_combined.png") if save_plots else None

    plot_xgb_shap_combined(
        explainer=explainer,
        shap_values=shap_values,
        X_test=X_test,
        sample_idx=sample_idx,
        title_prefix=title_suffix,
        save_path=combined_path,
        summary_max_display=8,
        bar_max_display=8,
        waterfall_max_display=8,
        shorten_names=True,
        name_max_len=30,
        show_feature_values=False,
        panel_dpi=160,
        display_figsize=(24, 7)
    )

    if save_csv:
        result["shap_importance_df"].to_csv(
            os.path.join(save_dir, "mean_abs_shap_ranking.csv"),
            index=False
        )

    print("\nTop 10 features by mean absolute SHAP:")
    print(result["shap_importance_df"].head(10))

    return result


def run_xgb_shap_batch(experiments, feature_configs, xgb_params,
                       save_root="results_featselect", sample_idx=0,
                       save_plots=True, background_size=100, save_csv=True):
    results = []

    for label, start_date in experiments:
        try:
            result = run_xgb_shap_analysis(
                label=label,
                start_date=start_date,
                feature_configs=feature_configs,
                xgb_params=xgb_params,
                save_root=save_root,
                sample_idx=sample_idx,
                save_plots=save_plots,
                background_size=background_size,
                save_csv=save_csv,
            )
            results.append(result)
        except Exception as e:
            print(f"Failed for {label} | {start_date}: {e}")

    return results


# ============================================================
# PERMUTATION IMPORTANCE ONLY
# ============================================================

def compute_xgb_permutation_importance(
    booster,
    X_test,
    y_test,
    metric="rmse",
    n_repeats=5,
    random_state=42
):
    """
    Permutation importance on a fixed fitted XGBoost Booster.
    No retraining. Only repeated prediction on shuffled copies of X_test.
    """
    rng = np.random.default_rng(random_state)

    if isinstance(y_test, pd.DataFrame):
        y_test = y_test.iloc[:, 0]
    if isinstance(y_test, pd.Series):
        y_test = y_test.values

    dtest = xgb.DMatrix(X_test)
    baseline_preds = booster.predict(dtest)

    if metric == "rmse":
        baseline_score = np.sqrt(np.mean((y_test - baseline_preds) ** 2))
    elif metric == "mae":
        baseline_score = np.mean(np.abs(y_test - baseline_preds))
    else:
        raise ValueError("metric must be 'rmse' or 'mae'")

    rows = []

    for col in X_test.columns:
        repeat_scores = []

        for _ in range(n_repeats):
            X_perm = X_test.copy()
            X_perm[col] = rng.permutation(X_perm[col].values)

            dperm = xgb.DMatrix(X_perm)
            perm_preds = booster.predict(dperm)

            if metric == "rmse":
                perm_score = np.sqrt(np.mean((y_test - perm_preds) ** 2))
            else:
                perm_score = np.mean(np.abs(y_test - perm_preds))

            repeat_scores.append(perm_score)

        mean_perm_score = float(np.mean(repeat_scores))
        std_perm_score = float(np.std(repeat_scores))
        importance = mean_perm_score - baseline_score

        rows.append({
            "feature": col,
            "baseline_score": float(baseline_score),
            "mean_permuted_score": mean_perm_score,
            "importance": float(importance),
            "std_across_repeats": std_perm_score
        })

    perm_importance_df = (
        pd.DataFrame(rows)
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    return perm_importance_df


def run_xgb_permutation_importance(label, start_date, feature_configs, xgb_params,
                                   save_root="results_featselect",
                                   metric="rmse", n_repeats=5, save_csv=True):
    result = load_xgb_experiment(
        label=label,
        start_date=start_date,
        feature_configs=feature_configs,
        xgb_params=xgb_params,
        save_root=save_root,
    )

    perm_importance_df = compute_xgb_permutation_importance(
        booster=result["booster"],
        X_test=result["X_test"],
        y_test=result["y_test"],
        metric=metric,
        n_repeats=n_repeats,
        random_state=42
    )

    result["perm_importance_df"] = perm_importance_df

    if save_csv:
        perm_importance_df.to_csv(
            os.path.join(result["save_dir"], "permutation_importance.csv"),
            index=False
        )

    print("\n" + "=" * 80)
    print(f"PERMUTATION IMPORTANCE | Label: {label} | Start date: {start_date} | Metric: {metric}")
    print("=" * 80)
    print(perm_importance_df.head(10))

    return result


def run_xgb_permutation_batch(experiments, feature_configs, xgb_params,
                              save_root="results_featselect",
                              metric="rmse", n_repeats=5, save_csv=True):
    results = []

    for label, start_date in experiments:
        try:
            result = run_xgb_permutation_importance(
                label=label,
                start_date=start_date,
                feature_configs=feature_configs,
                xgb_params=xgb_params,
                save_root=save_root,
                metric=metric,
                n_repeats=n_repeats,
                save_csv=save_csv,
            )
            results.append(result)
        except Exception as e:
            print(f"Failed for {label} | {start_date}: {e}")

    return results