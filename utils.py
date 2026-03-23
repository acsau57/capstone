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

def rebuild_xgb_splits_for_shap(args):
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

def explain_xgb_model(label, start_date, feature_configs, xgb_params,
                      save_root="results_featselect", background_size=100):
    """
    Load one trained XGBoost model and compute SHAP values on its test set.
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

    X_train, X_val, X_test, y_df = rebuild_xgb_splits_for_shap(args)

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

    background = X_train.sample(min(background_size, len(X_train)), random_state=42)

    explainer = shap.TreeExplainer(booster, data=background)
    shap_values = explainer.shap_values(X_test)

    return {
        "label": label,
        "start_date": start_date,
        "save_dir": save_dir,
        "explainer": explainer,
        "shap_values": shap_values,
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_df": y_df,
    }

def plot_xgb_shap_summary(shap_values, X_test, title=None, save_path=None):
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, show=False)
    if title:
        plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_xgb_shap_bar(shap_values, X_test, title=None, save_path=None):
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    if title:
        plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_xgb_shap_waterfall(explainer, shap_values, X_test, sample_idx=0,
                            title=None, save_path=None, max_display=15):
    explanation = shap.Explanation(
        values=shap_values[sample_idx],
        base_values=explainer.expected_value,
        data=X_test.iloc[sample_idx],
        feature_names=X_test.columns.tolist()
    )

    plt.figure(figsize=(10, 8))
    shap.plots.waterfall(explanation, max_display=max_display, show=False)
    if title:
        plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

def run_xgb_shap_analysis(label, start_date, feature_configs, xgb_params,
                          save_root="results_featselect", sample_idx=0,
                          save_plots=True, background_size=100):
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
        save_path=os.path.join(save_dir, "shap_summary.png") if save_plots else None
    )

    plot_xgb_shap_bar(
        shap_values,
        X_test,
        title=f"SHAP Bar - {title_suffix}",
        save_path=os.path.join(save_dir, "shap_bar.png") if save_plots else None
    )

    plot_xgb_shap_waterfall(
        explainer,
        shap_values,
        X_test,
        sample_idx=sample_idx,
        title=f"SHAP Waterfall - {title_suffix} - sample {sample_idx}",
        save_path=os.path.join(save_dir, f"shap_waterfall_{sample_idx}.png") if save_plots else None
    )

    return result

def run_xgb_shap_batch(experiments, feature_configs, xgb_params,
                       save_root="results_featselect", sample_idx=0,
                       save_plots=True, background_size=100):
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
            )
            results.append(result)
        except Exception as e:
            print(f"Failed for {label} | {start_date}: {e}")

    return results


    

# def explain_model(model, data_loader, args, num_samples=100):
#     """
#     Generate SHAP explanations using KernelExplainer
#     (Model-agnostic)
#     """
#     model.eval()
    
#     # Get feature names - BUILD COMPLETE LIST
#     feature_names = []
    
#     # 1. Base features
#     if hasattr(args, 'features'):
#         feature_names.extend(args.features)
    
#     # 2. Lag features
#     if hasattr(args, 'labels') and hasattr(args, 'lag_periods'):
#         for label in args.labels:
#             for lag in args.lag_periods:
#                 feature_names.append(f'{label}_lag_{lag}')
    
#     # 3. Dummy variables
#     if hasattr(args, 'dummy_vars'):
#         feature_names.extend(args.dummy_vars)
    
#     # 4. Seasonal features (if enabled)
#     use_seasonal = getattr(args, 'use_seasonal', False)
#     if use_seasonal:
#         seasonal_features = ['month_sin', 'month_cos', 'quarter_sin', 'quarter_cos', 
#                            'is_tax_season', 'is_year_end']
#         feature_names.extend(seasonal_features)
    
#     # Collect background and test data
#     background_data = []
#     test_data = []
    
#     for i, (inputs, _) in enumerate(data_loader):
#         # Extract last timestep from sequences
#         last_step = inputs[:, -1, :].cpu().numpy()
        
#         if i == 0:
#             background_data = last_step[:num_samples]
        
#         test_data.append(last_step)
        
#         if len(test_data) * inputs.shape[0] >= 20:
#             break
    
#     test_data = np.vstack(test_data)[:20]  # Use first 20 samples

    
#     # Create prediction wrapper for SHAP
#     def model_predict(x):
#         """Wrapper function that takes 2D array and returns predictions"""
#         x_tensor = torch.FloatTensor(x).to(args.device)
#         x_tensor = x_tensor.unsqueeze(1)
        
#         with torch.no_grad():
#             output = model(x_tensor).cpu().numpy()
        
#         return output
    
#     # Create KernelExplainer
#     explainer = shap.KernelExplainer(model_predict, background_data)
    
#     # Calculate SHAP values
#     shap_values = explainer.shap_values(test_data, nsamples=100)

#     return explainer, shap_values, test_data, feature_names


# def plot_shap_summary(shap_values, test_data, feature_names, output_idx=0, output_name=None):
#     """Plot SHAP summary - shows feature importance"""
    
#     # For multi-output models
#     n_outputs = shap_values.shape[1] // len(feature_names) if len(shap_values.shape) == 2 else 1
#     n_features = len(feature_names)
    
#     if n_outputs > 1:
#         # Reshape and extract specific output
#         shap_values_reshaped = shap_values.reshape(shap_values.shape[0], n_features, n_outputs)
#         values = shap_values_reshaped[:, :, output_idx]
#     else:
#         values = shap_values
    
#     title = f'SHAP Feature Importance - {output_name}' if output_name else f'SHAP Feature Importance - Output {output_idx}'
    
#     plt.figure(figsize=(10, 8))
#     shap.summary_plot(values, test_data, feature_names=feature_names, show=False)
#     plt.title(title)
#     plt.tight_layout()
#     plt.show()


# def plot_shap_bar(shap_values, test_data, feature_names, output_idx=0, output_name=None):
#     """Plot SHAP bar chart - mean absolute SHAP values"""
    
#     # For multi-output models
#     n_outputs = shap_values.shape[1] // len(feature_names) if len(shap_values.shape) == 2 else 1
#     n_features = len(feature_names)
    
#     if n_outputs > 1:
#         # Reshape and extract specific output
#         shap_values_reshaped = shap_values.reshape(shap_values.shape[0], n_features, n_outputs)
#         values = shap_values_reshaped[:, :, output_idx]
#     else:
#         values = shap_values
    
#     title = f'SHAP Mean Importance - {output_name}' if output_name else f'SHAP Mean Importance - Output {output_idx}'
    
#     plt.figure(figsize=(10, 6))
#     shap.summary_plot(values, test_data, feature_names=feature_names, plot_type="bar", show=False)
#     plt.title(title)
#     plt.tight_layout()
#     plt.show()


# def plot_shap_waterfall(explainer, shap_values, test_data, feature_names, sample_idx=0, output_idx=0, output_name=None):
#     """Plot SHAP waterfall - explains a single prediction"""
    
#     # For multi-output models, SHAP concatenates outputs
#     # Shape is (samples, features * n_outputs)
#     n_outputs = len(explainer.expected_value) if hasattr(explainer.expected_value, '__len__') else 1
#     n_features = len(feature_names)
    
#     if n_outputs > 1:
#         # Reshape from (samples, features * outputs) to (samples, features, outputs)
#         shap_values_reshaped = shap_values.reshape(shap_values.shape[0], n_features, n_outputs)
#         values = shap_values_reshaped[sample_idx, :, output_idx]
#         base_value = explainer.expected_value[output_idx]
#         data = test_data[sample_idx]
#     else:
#         values = shap_values[sample_idx]
#         base_value = explainer.expected_value
#         data = test_data[sample_idx]
    
#     # Create explanation object
#     explanation = shap.Explanation(
#         values=values,
#         base_values=base_value,
#         data=data,
#         feature_names=feature_names
#     )
    
#     title = f'SHAP Waterfall - Sample {sample_idx}, {output_name}' if output_name else f'SHAP Waterfall - Sample {sample_idx}, Output {output_idx}'
    
#     plt.figure(figsize=(10, 8))
#     shap.waterfall_plot(explanation, show=False)
#     plt.title(title)
#     plt.tight_layout()
#     plt.show()


# def plot_shap_dependence(shap_values, test_data, feature_names, feature_idx, output_idx=0, output_name=None):
#     """Plot SHAP dependence plot for a specific feature"""
    
#     # Handle multi-output case
#     if isinstance(shap_values, list):
#         values = shap_values[output_idx]
#     else:
#         values = shap_values
    
#     feature_name = feature_names[feature_idx]
#     title = f'SHAP Dependence: {feature_name} - {output_name}' if output_name else f'SHAP Dependence: {feature_name}'
    
#     plt.figure(figsize=(10, 6))
#     shap.dependence_plot(feature_idx, values, test_data, feature_names=feature_names, show=False)
#     plt.title(title)
#     plt.tight_layout()
#     plt.show()