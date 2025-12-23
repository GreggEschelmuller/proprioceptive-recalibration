import numpy as np
import pandas as pd


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    n_boot: int = 10_000,
    ci: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """
    Bootstrap CI for the mean of `values`.

    values: 1D array (NaNs allowed; dropped internally)
    """
    if rng is None:
        rng = np.random.default_rng()

    x = values[~np.isnan(values)]
    n = len(x)

    if n < 2:
        return np.nan, np.nan

    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(x, size=n, replace=True)
        boot_means[i] = sample.mean()

    alpha = (1 - ci) / 2
    lo = np.quantile(boot_means, alpha)
    hi = np.quantile(boot_means, 1 - alpha)

    return lo, hi

def summarize_mean_ci_by_trial_bin(
    df: pd.DataFrame,
    *,
    error_col: str = "error",
    trial_col: str = "trial",
    block_col: str = "block",
    participant_col: str = "participant",
    block_order: list[str],
    trials_per_block: int = 100,
    bin_size: int = 5,
    n_boot: int = 10_000,
    ci_level: float = 0.95,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Compute mean error and bootstrap CIs across participants for binned trials.

    Bins are non-overlapping windows on the concatenated trial axis:
      e.g., if bin_size=5 => 1–5, 6–10, 11–15, ...

    Returns columns:
      - _trial_bin (int)          # bin index on concatenated axis (1-based bins)
      - _global_trial_center (float)  # x-value to plot (center of the bin)
      - <block_col>
      - mean
      - ci_lo
      - ci_hi
      - n                         # participants contributing (non-NaN bin means)
    """
    # ---- validation ----
    for col in (error_col, trial_col, block_col, participant_col):
        if col not in df.columns:
            raise KeyError(f"Missing required column '{col}'")

    if not block_order:
        raise ValueError("block_order must be non-empty")

    if bin_size <= 0:
        raise ValueError("bin_size must be > 0")

    # ---- filter blocks + enforce order ----
    work = df[df[block_col].isin(block_order)].copy()
    work[block_col] = pd.Categorical(work[block_col], categories=block_order, ordered=True)

    # ---- build concatenated trial axis ----
    block_to_idx = {b: i for i, b in enumerate(block_order)}
    work["_block_idx"] = work[block_col].map(block_to_idx).astype("int64")

    work[trial_col] = pd.to_numeric(work[trial_col], errors="raise").astype("int64")
    work["_global_trial"] = work["_block_idx"] * trials_per_block + work[trial_col]

    # ---- define non-overlapping bins on the concatenated axis ----
    # If trials are 1-based: 1..5 -> bin 1, 6..10 -> bin 2, ...
    work["_trial_bin"] = ((work["_global_trial"] - 1) // bin_size + 1).astype("int64")

    # x-coordinate (center of the bin)
    work["_global_trial_center"] = (work["_trial_bin"] - 1) * bin_size + (bin_size + 1) / 2

    # ---- within each participant and bin, average errors ----
    pt_bin = (
        work.groupby(
            [participant_col, "_trial_bin", "_global_trial_center", block_col],
            observed=True,
        )[error_col]
        .mean()  # mean within participant across trials in the bin (NaNs ignored)
        .reset_index()
    )

    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    # ---- across participants, compute mean + bootstrap CI per bin ----
    for (tbin, center, block), sub in pt_bin.groupby(
        ["_trial_bin", "_global_trial_center", block_col],
        observed=True,
    ):
        values = sub[error_col].to_numpy(dtype=float)

        mean = np.nanmean(values)
        ci_lo, ci_hi = bootstrap_mean_ci(
            values,
            n_boot=n_boot,
            ci=ci_level,
            rng=rng,
        )

        rows.append(
            {
                "_trial_bin": int(tbin),
                "_global_trial_center": float(center),
                block_col: block,
                "mean": float(mean),
                "ci_lo": float(ci_lo),
                "ci_hi": float(ci_hi),
                "n": int(np.sum(~np.isnan(values))),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("_global_trial_center")
        .reset_index(drop=True)
    )