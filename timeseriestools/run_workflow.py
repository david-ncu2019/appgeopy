"""
Workflow for SSA-based Imputation & Smoothing.

This script uses the robust functions from 'pca_imputation.py'.
It is designed to:
- Extract the main components of the time series (trend + large-scale signal).
- Produce a smoother reconstructed series that does NOT track high-frequency noise.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

# Import SSA functions - handle both relative and absolute imports
try:
    from .pca_imputation import impute_ssa, suggest_parameters
except ImportError:
    try:
        from pca_imputation import impute_ssa, suggest_parameters
    except ImportError:
        print("Error: Could not import from 'pca_imputation.py'.")
        print("Please ensure the file is in the same directory.")
        raise


def _calculate_imputation_metrics(original_values, imputed_values):
    """
    Helper to calculate standard metrics on known values only.
    """
    valid_mask = ~np.isnan(original_values) & ~np.isnan(imputed_values)
    if np.sum(valid_mask) < 2:
        return {"rmse": np.nan, "mae": np.nan, "r2": np.nan}

    orig_valid = original_values[valid_mask]
    imp_valid = imputed_values[valid_mask]

    rmse = np.sqrt(mean_squared_error(orig_valid, imp_valid))
    mae = np.mean(np.abs(orig_valid - imp_valid))
    r2 = r2_score(orig_valid, imp_valid)

    return {"rmse": rmse, "mae": mae, "r2": r2}


def validate_imputation_stability(
    data,
    embedding_dim,
    n_components=None,
    variance_threshold=0.9,
    max_components=None,
    time_col=None,
    value_col=None,
    mask_ratio=0.1,
    random_seed=42,
    max_iter=30,
    smooth_observed=False,
):
    """
    Validate imputation by masking random known values and checking
    how well the method recovers them (quick sanity check).
    """
    # 1. Extract series values
    if isinstance(data, pd.Series):
        series_values = data.values
        original_index = data.index
    elif isinstance(data, pd.DataFrame):
        if value_col is None:
            raise ValueError("`value_col` must be provided for a DataFrame.")
        series_values = data[value_col].values
        original_index = data.index
    else:
        series_values = np.array(data).flatten()
        original_index = None

    # 2. Artificially mask some values
    np.random.seed(random_seed)
    valid_indices = np.where(~np.isnan(series_values))[0]
    n_to_mask = int(len(valid_indices) * mask_ratio)
    if n_to_mask == 0:
        return {"rmse": 0, "mae": 0, "r2": 1}, [], [], []

    mask_indices = np.random.choice(valid_indices, size=n_to_mask, replace=False)
    masked_series = series_values.copy()
    masked_series[mask_indices] = np.nan

    if original_index is not None:
        masked_data_input = pd.Series(masked_series, index=original_index)
    else:
        masked_data_input = masked_series

    # 3. Run SSA imputation on the masked data
    imputed_series = impute_ssa(
        masked_data_input,
        embedding_dim=embedding_dim,
        n_components=n_components,
        variance_threshold=variance_threshold,
        max_components=max_components,
        time_col=time_col,
        value_col=value_col,
        max_iter=max_iter,
        smooth_observed=smooth_observed,
    )

    # 4. Extract imputed values at masked locations
    if isinstance(imputed_series, pd.Series):
        imputed_values_at_mask = imputed_series.values[mask_indices]
    else:
        imputed_values_at_mask = imputed_series[mask_indices]
    original_values_at_mask = series_values[mask_indices]

    # 5. Calculate metrics
    metrics = _calculate_imputation_metrics(
        original_values_at_mask, imputed_values_at_mask
    )

    return metrics, original_values_at_mask, imputed_values_at_mask, mask_indices


def run_ssa_smoothing_workflow(
    data,
    time_col=None,
    value_col=None,
    fixed_n_components=None,
    variance_threshold=0.9,
    max_components=None,
    smooth_observed=True,  # NEW: Enable full smoothing by default
):
    """
    Execute SSA smoothing & imputation workflow.

    Key design:
    - First, auto-tune `embedding_dim` from the data.
    - Then, perform SSA with either:
        * fixed_n_components: a small r (e.g., 2–4) to get very smooth trend, OR
        * variance_threshold: keep enough components to explain a given
          fraction of variance, with an optional `max_components` cap.
    - smooth_observed: If True, apply smoothing to entire series (denoising).
      If False, only fill missing values.

    Recommended usage:
    - For very smooth trend: fixed_n_components=2 or 3, variance_threshold=0.8
    - For moderate smoothing: fixed_n_components=None, variance_threshold=0.9,
      max_components=5
    """
    # 1. Extract series values for parameter suggestion
    if isinstance(data, pd.Series):
        series_values = data.values
    elif isinstance(data, pd.DataFrame):
        if value_col is None:
            # Try to infer a single value column
            cols = [c for c in data.columns if c != time_col]
            if len(cols) == 1:
                value_col = cols[0]
            else:
                raise ValueError("For DataFrame input, `value_col` must be specified.")
        series_values = data[value_col].values
    else:
        series_values = np.array(data).flatten()

    # 2. Auto-tune embedding dimension
    try:
        best_embedding_dim = suggest_parameters(series_values)
        print(f"Auto-tuned embedding_dim (L): {best_embedding_dim}")
    except Exception as e:
        print(f"Warning: Failed to auto-tune embedding_dim. Falling back. Error: {e}")
        best_embedding_dim = max(2, len(series_values) // 10)
        print(f"Fallback embedding_dim (L): {best_embedding_dim}")

    # 3. Decide on component selection mode
    if fixed_n_components is not None:
        # Explicit low-rank smoothing
        chosen_n_components = int(fixed_n_components)
        chosen_variance_threshold = variance_threshold  # still used if r is large
        chosen_max_components = max_components
        print(f"Using fixed_n_components={chosen_n_components}")
    else:
        # Use variance-based selection
        chosen_n_components = None
        chosen_variance_threshold = variance_threshold
        chosen_max_components = max_components
        print(
            f"Using variance_threshold={chosen_variance_threshold}, "
            f"max_components={chosen_max_components}"
        )

    print(f"Smoothing mode: {'Full smoothing (denoise ALL values)' if smooth_observed else 'Gap filling only'}")

    # 4. Run final SSA smoothing + imputation
    imputed_series = impute_ssa(
        data=data,
        embedding_dim=best_embedding_dim,
        n_components=chosen_n_components,
        variance_threshold=chosen_variance_threshold,
        max_components=chosen_max_components,
        time_col=time_col,
        value_col=value_col,
        max_iter=50,
        smooth_observed=smooth_observed,
    )

    # 5. Optional: simple validation (can be turned off if not needed)
    try:
        mask_ratio = min(0.1, 0.5 * (1 - np.isnan(series_values).mean()))
        validation_metrics, _, _, _ = validate_imputation_stability(
            data=data,
            embedding_dim=best_embedding_dim,
            n_components=chosen_n_components,
            variance_threshold=chosen_variance_threshold,
            max_components=chosen_max_components,
            time_col=time_col,
            value_col=value_col,
            mask_ratio=mask_ratio,
            random_seed=43,
            max_iter=30,
            smooth_observed=smooth_observed,
        )
    except Exception as e:
        print(f"Error during validation: {e}")
        validation_metrics = {"rmse": np.nan, "mae": np.nan, "r2": np.nan}

    return {
        "imputed_series": imputed_series,
        "parameters": {
            "embedding_dim": best_embedding_dim,
            "n_components": chosen_n_components,
            "variance_threshold": chosen_variance_threshold,
            "max_components": chosen_max_components,
            "smooth_observed": smooth_observed,
        },
        "validation_metrics": validation_metrics,
    }


# --- Example Usage ---
if __name__ == "__main__":
    np.random.seed(42)
    # 1. Create a synthetic dummy dataset (pd.Series with DatetimeIndex)
    n = 300
    t = np.arange(n)

    # Smooth trend + low-frequency oscillation
    y_true = np.sin(t * 0.1) + 0.5 * np.sin(t * 0.05) + (t * 0.02)
    # Add noise
    y_noise = y_true + np.random.normal(0, 0.2, n)

    # Introduce missing values
    y_missing = y_noise.copy()
    y_missing[50:80] = np.nan  # Block gap
    y_missing[np.random.choice(n, 30, replace=False)] = np.nan  # Random gaps

    date_index = pd.date_range(start="2024-01-01", periods=n, freq="D")
    series_data = pd.Series(y_missing, index=date_index)

    print("Created synthetic data for SSA smoothing test.")
    print(f"Total points: {n}")
    print(f"Missing: {series_data.isna().sum()} points")

    # 2. Run SSA smoothing workflow
    # Example: very smooth trend with 3 components
    results = run_ssa_smoothing_workflow(
        data=series_data,
        fixed_n_components=None,       # control smoothness here
        variance_threshold=0.95,    # used only if fixed_n_components=None
        max_components=5,           # optional cap
        smooth_observed=True,       # CRITICAL: Enable full smoothing!
    )

    print("\n--- WORKFLOW COMPLETE ---")
    print("Parameters used:", results["parameters"])
    print("Validation metrics:", results["validation_metrics"])

    # 3. Plot (optional)
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(15, 6))
        pd.Series(y_true, index=date_index).plot(
            label="True underlying signal", style="k--", alpha=0.4
        )
        pd.Series(y_noise, index=date_index).plot(
            label="Noisy observed (no gaps)", style="gray", alpha=0.3
        )
        series_data.plot(
            label="Observed with gaps", style="b.", alpha=0.6, markersize=5
        )
        results["imputed_series"].plot(
            label="SSA smoothed & imputed", style="r-", linewidth=2
        )

        plt.title("Iterative SSA Smoothing & Imputation")
        plt.legend()
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("Matplotlib not found. Skipping plot.")
