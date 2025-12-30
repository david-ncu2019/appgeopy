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

# Import SSA functions
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

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # mask_indices = np.random.choice(valid_indices, size=n_to_mask, replace=False)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # Ensure we mask at least 5% for meaningful validation
    n_to_mask = max(n_to_mask, int(len(valid_indices) * 0.05))
    n_to_mask = min(n_to_mask, len(valid_indices))
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
    embedding_dim=None,  # ADD THIS
    fixed_n_components=None,
    variance_threshold=0.9,
    max_components=None,
    max_iter=50,  # ADD THIS
    tol=1e-5,  # ADD THIS
    smooth_observed=True,
    verbose=True,  # ADD THIS
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

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # # 2. Auto-tune embedding dimension
    # try:
    #     best_embedding_dim = suggest_parameters(series_values)
    #     print(f"Auto-tuned embedding_dim (L): {best_embedding_dim}")
    # except Exception as e:
    #     print(f"Warning: Failed to auto-tune embedding_dim. Falling back. Error: {e}")
    #     best_embedding_dim = max(2, len(series_values) // 10)
    #     print(f"Fallback embedding_dim (L): {best_embedding_dim}")
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    # 2. Optimize or auto-tune parameters
    if embedding_dim is None and fixed_n_components is None:
        # Statistical optimization via cross-validation
        if verbose:
            print("Running parameter optimization via CV...")
        from pca_imputation import optimize_ssa_parameters
        
        optimal = optimize_ssa_parameters(
            series_values,
            n_folds=5,
            metric='rmse'
        )
        
        best_embedding_dim = optimal['embedding_dim']
        chosen_n_components = optimal['n_components']
        
        if verbose:
            print(f"Optimized: embedding_dim={best_embedding_dim}, "
                  f"n_components={chosen_n_components}, CV_RMSE={optimal['cv_score']:.3f}")
    else:
        # Use provided or auto-tune (existing logic)
        if embedding_dim is None:
            try:
                best_embedding_dim = suggest_parameters(series_values)
                if verbose:
                    print(f"Auto-tuned embedding_dim (L): {best_embedding_dim}")
            except Exception as e:
                if verbose:
                    print(f"Warning: Failed to auto-tune. Falling back. Error: {e}")
                best_embedding_dim = max(2, len(series_values) // 10)
                if verbose:
                    print(f"Fallback embedding_dim (L): {best_embedding_dim}")
        else:
            best_embedding_dim = embedding_dim
            if verbose:
                print(f"Using provided embedding_dim (L): {best_embedding_dim}")
        
        chosen_n_components = fixed_n_components  # Will be set in section 3

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # 3. Decide on component selection mode
    # if fixed_n_components is not None:
    #     # Explicit low-rank smoothing
    #     chosen_n_components = int(fixed_n_components)
    #     chosen_variance_threshold = variance_threshold  # still used if r is large
    #     chosen_max_components = max_components
    #     print(f"Using fixed_n_components={chosen_n_components}")
    # else:
    #     # Use variance-based selection
    #     chosen_n_components = None
    #     chosen_variance_threshold = variance_threshold
    #     chosen_max_components = max_components
    #     print(
    #         f"Using variance_threshold={chosen_variance_threshold}, "
    #         f"max_components={chosen_max_components}"
    #     )

    # print(f"Smoothing mode: {'Full smoothing (denoise ALL values)' if smooth_observed else 'Gap filling only'}")
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # # 3. Decide on component selection mode
    # if fixed_n_components is not None:
    #     chosen_n_components = int(fixed_n_components)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # 3. Decide on component selection mode (if not already optimized)
    if chosen_n_components is None:  # Not set by optimizer
        if fixed_n_components is not None:
            chosen_n_components = int(fixed_n_components)
        else:
            chosen_n_components = None  # Use variance threshold
        
        chosen_variance_threshold = variance_threshold
        chosen_max_components = max_components
        
        if verbose:
            if chosen_n_components is not None:
                print(f"Using fixed_n_components={chosen_n_components}")
            else:
                print(
                    f"Using variance_threshold={chosen_variance_threshold}, "
                    f"max_components={chosen_max_components}"
                )
    else:
        # Already optimized - set other params
        chosen_variance_threshold = variance_threshold
        chosen_max_components = max_components

    if verbose:
        print(f"Smoothing mode: {'Full smoothing (denoise ALL values)' if smooth_observed else 'Gap filling only'}")

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # # 4. Run final SSA smoothing + imputation
    # imputed_series = impute_ssa(
    #     data=data,
    #     embedding_dim=best_embedding_dim,
    #     n_components=chosen_n_components,
    #     variance_threshold=chosen_variance_threshold,
    #     max_components=chosen_max_components,
    #     time_col=time_col,
    #     value_col=value_col,
    #     max_iter=50,
    #     smooth_observed=smooth_observed,
    # )
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    imputed_series = impute_ssa(
            data=data,
            embedding_dim=best_embedding_dim,
            n_components=chosen_n_components,
            variance_threshold=chosen_variance_threshold,
            max_components=chosen_max_components,
            time_col=time_col,
            value_col=value_col,
            max_iter=max_iter,
            tol=tol,
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
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # except Exception as e:
    #     print(f"Error during validation: {e}")
    #     validation_metrics = {"rmse": np.nan, "mae": np.nan, "r2": np.nan}
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    except (ValueError, RuntimeError) as e:
        if verbose:
            print(f"Warning: Validation failed: {e}")
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
    import matplotlib.pyplot as plt
    
    np.random.seed(42)
    
    # 1. Generate synthetic data with multiple characteristics
    n = 300
    t = np.arange(n)
    
    # Components: trend + two seasonal patterns + noise
    trend = 0.02 * t
    slow_season = 3 * np.sin(2 * np.pi * t / 80)
    fast_season = 1 * np.sin(2 * np.pi * t / 20)
    noise = np.random.normal(0, 0.3, n)
    
    y_true = trend + slow_season + fast_season
    y_observed = y_true + noise
    
    # Create realistic missing patterns
    y_missing = y_observed.copy()
    y_missing[50:80] = np.nan      # Block gap 1
    y_missing[150:170] = np.nan    # Block gap 2
    y_missing[250:265] = np.nan    # Block gap 3
    # Random scattered gaps (20% additional)
    random_gaps = np.random.choice(n, size=int(n * 0.2), replace=False)
    y_missing[random_gaps] = np.nan
    
    date_index = pd.date_range(start="2024-01-01", periods=n, freq="D")
    series_data = pd.Series(y_missing, index=date_index)
    
    missing_pct = series_data.isna().sum() / n * 100
    print("="*60)
    print("SYNTHETIC DATA GENERATED")
    print("="*60)
    print(f"Total points: {n}")
    print(f"Missing: {series_data.isna().sum()} ({missing_pct:.1f}%)")
    print(f"Observed range: [{series_data.min():.2f}, {series_data.max():.2f}]")
    print("="*60)
    
    # 2. Test with different configurations
    configs = [
        {
            "name": "Method 1: Auto-Optimized (CV)",
            "params": {
                "smooth_observed": False,  # Will auto-optimize both params
                "verbose": True
            }
        },
        {
            "name": "Method 2: Manual Fixed",
            "params": {
                "embedding_dim": 40,
                "fixed_n_components": 5,
                "smooth_observed": False,
                "verbose": True
            }
        },
        {
            "name": "Method 3: Auto embedding_dim only",
            "params": {
                "fixed_n_components": 3,  # Smooth extraction
                "smooth_observed": True,
                "verbose": True
            }
        }
    ]
    
    results_list = []
    
    for config in configs:
        print(f"\nTesting: {config['name']}")
        result = run_ssa_smoothing_workflow(
            data=series_data,
            **config['params']
        )
        results_list.append((config['name'], result))
        
        # Calculate accuracy on observed points
        observed_mask = ~np.isnan(y_observed)
        rmse_obs = np.sqrt(np.mean(
            (result['imputed_series'].values[observed_mask] - y_true[observed_mask])**2
        ))
        print(f"  RMSE vs true signal: {rmse_obs:.3f}")
        print(f"  Validation metrics: {result['validation_metrics']}")
    
    # 3. Visualize comparison
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    for ax, (name, result) in zip(axes, results_list):
        # Plot data
        ax.plot(date_index, y_true, 'k-', linewidth=1.5, 
                alpha=0.4, label='True signal')
        ax.plot(date_index, y_observed, 'gray', linewidth=0.5, 
                alpha=0.3, label='Noisy observed')
        ax.plot(date_index, series_data, 'b.', markersize=4, 
                alpha=0.6, label='With gaps')
        ax.plot(date_index, result['imputed_series'], 'r-', 
                linewidth=2, alpha=0.8, label='Imputed')
        
        # Formatting
        ax.set_title(f"{name} | " + 
                    f"embedding_dim={result['parameters']['embedding_dim']}, " +
                    f"n_components={result['parameters']['n_components']}", 
                    fontsize=11, fontweight='bold')
        ax.set_ylabel('Value')
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Date')
    plt.tight_layout()
    plt.savefig('ssa_workflow_test.png', dpi=150, bbox_inches='tight')
    print("\n" + "="*60)
    print("Plot saved: ssa_workflow_test.png")
    print("="*60)
    plt.show()
