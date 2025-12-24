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


def prepare_binned_data(
        df: pd.DataFrame,
        *,
        trial_col: str,
        block_col: str,
        participant_col: str,
        block_order: list[str],
        trials_per_block: int,
        bin_size: int,
) -> pd.DataFrame:
    for col in (trial_col, block_col, participant_col):
        if col not in df.columns:
            raise KeyError(f"Missing required column '{col}'")
    if not block_order:
        raise ValueError("block_order must be non-empty")
    if bin_size <= 0:
        raise ValueError("bin_size must be > 0")
    
    work = df[df[block_col].isin(block_order)].copy()
    work[block_col] = pd.Categorical(work[block_col], categories=block_order, ordered=True)

    block_to_idx = {b: i for i, b in enumerate(block_order)}
    work["_block_idx"] = work[block_col].map(block_to_idx).astype("int64")

    work[trial_col] = pd.to_numeric(work[trial_col], errors="raise").astype("int64")
    work["_global_trial"] = work["_block_idx"] * trials_per_block + work[trial_col]

    work["_trial_bin"] = ((work["_global_trial"] - 1) // bin_size + 1).astype("int64")
    work["_global_trial_center"] = (work["_trial_bin"] - 1) * bin_size + (bin_size + 1) / 2

    return work

def summarize_value_with_boot_ci(
    pt: pd.DataFrame,
    *,
    value_col: str,
    block_col: str,
    n_boot: int,
    ci_level: float,
    seed: int,
    out_prefix: str,
) -> pd.DataFrame:
    required = {"_trial_bin", "_global_trial_center", block_col, value_col}
    missing = required - set(pt.columns)
    if missing:
        raise KeyError(f"pt missing columns: {sorted(missing)}")

    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    for (tbin, center, block), sub in pt.groupby(
        ["_trial_bin", "_global_trial_center", block_col],
        observed=True,
    ):
        vals = sub[value_col].to_numpy(dtype=float)
        n = int(np.sum(~np.isnan(vals)))
        if n == 0:
            continue  # nothing to summarize for this bin

        m = float(np.nanmean(vals))
        lo, hi = bootstrap_mean_ci(vals, n_boot=n_boot, ci=ci_level, rng=rng)

        rows.append(
            {
                "_trial_bin": int(tbin),
                "_global_trial_center": float(center),
                block_col: block,
                f"{out_prefix}_mean": m,
                f"{out_prefix}_ci_lo": float(lo),
                f"{out_prefix}_ci_hi": float(hi),
                f"n_{out_prefix}": n,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("_global_trial_center")
        .reset_index(drop=True)
    )

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
    if error_col not in df.columns:
        raise KeyError(f"Missing required column {error_col}")

    work = prepare_binned_data(
        df,
        trial_col=trial_col,
        block_col=block_col,
        participant_col=participant_col,
        block_order=block_order,
        trials_per_block=trials_per_block,
        bin_size=bin_size,
    )

    pt = (
        work.groupby([participant_col, "_trial_bin", "_global_trial_center", block_col], observed=True)[error_col]
        .mean()
        .rename("value")
        .reset_index()
    )

    return summarize_value_with_boot_ci(
        pt,
        value_col="value",
        block_col=block_col,
        n_boot=n_boot,
        ci_level=ci_level,
        seed=seed,
        out_prefix="mean",
    )

def summarize_within_subject_ve_ci_by_trial_bin(
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
    if error_col not in df.columns:
        raise KeyError(f"Missing required column '{error_col}'")

    work = prepare_binned_data(
        df,
        trial_col=trial_col,
        block_col=block_col,
        participant_col=participant_col,
        block_order=block_order,
        trials_per_block=trials_per_block,
        bin_size=bin_size,
    )

    pt = (
        work.groupby([participant_col, "_trial_bin", "_global_trial_center", block_col], observed=True)[error_col]
        .std()  # within-subject SD across trials in bin
        .rename("value")
        .reset_index()
    )

    return summarize_value_with_boot_ci(
        pt,
        value_col="value",
        block_col=block_col,
        n_boot=n_boot,
        ci_level=ci_level,
        seed=seed,
        out_prefix="ve",
    )


def mean_first_n_trials(
    df: pd.DataFrame,
    *,
    value_col: str,
    trial_col: str,
    n: int = 5,
    min_valid: int = 3,
) -> float:
    """
    Mean of the first n trials (by trial_col).
    NaNs ignored; returns NaN if fewer than min_valid remain.
    """
    for col in (trial_col, value_col):
        if col not in df.columns:
            raise KeyError(f"Missing required column '{col}'")

    if df.empty:
        msg = (
            f"No trials available for participant={participant}, block={block}"
        )
        warnings.warn(msg, RuntimeWarning)
        return np.nan

    work = df.sort_values(trial_col)
    data = work.head(n)[value_col].to_numpy(dtype=float)
    values = values[~np.isnan(values)]

    if len(values) < min_valid:
        msg = (
            f"Insufficient valid trials ({len(values)}/{min_valid}) "
            f"for participant={participant}, block={block}"
        )
        warnings.warn(msg, RuntimeWarning)
        return np.nan

    return data_mean


def summarize_recalibration_two_blocks(
    df: pd.DataFrame,
    *,
    participant_col: str = "participant",
    block_col: str = "block",
    trial_col: str = "trial_num",
    value_col: str = "error",
    block_a: str,
    block_b: str,
    n: int = 5,
    min_valid: int = 5,
    diff_name: str = "diff",
) -> pd.DataFrame:
    """
    Participant-level summary comparing the first N trials of two blocks.

    Returns one row per participant:
      participant, a_mean, b_mean, diff (= b - a)

    Notes:
    - NaNs propagate: if a_mean or b_mean is NaN, diff is NaN.
    - Warnings about missing/insufficient trials should come from mean_first_n_trials().
    """
    # ---- validation ----
    for col in (participant_col, block_col, trial_col, value_col):
        if col not in df.columns:
            raise KeyError(f"Missing required column '{col}'")

    if block_a == block_b:
        raise ValueError("block_a and block_b must be different")

    rows: list[dict] = []

    for pid, df_p in df.groupby(participant_col, sort=True):
        df_a = df_p[df_p[block_col] == block_a]
        df_b = df_p[df_p[block_col] == block_b]

        a_mean = mean_first_n_trials(
            df_a,
            value_col=value_col,
            trial_col=trial_col,
            n=n,
            min_valid=min_valid,
            participant=int(pid) if pd.notna(pid) else None,
            block=block_a,
        )

        b_mean = mean_first_n_trials(
            df_b,
            value_col=value_col,
            trial_col=trial_col,
            n=n,
            min_valid=min_valid,
            participant=int(pid) if pd.notna(pid) else None,
            block=block_b,
        )

        diff = b_mean - a_mean if (np.isfinite(a_mean) and np.isfinite(b_mean)) else np.nan

        rows.append(
            {
                participant_col: pid,
                "block_a": block_a,
                "block_b": block_b,
                "a_mean": float(a_mean) if np.isfinite(a_mean) else np.nan,
                "b_mean": float(b_mean) if np.isfinite(b_mean) else np.nan,
                diff_name: float(diff) if np.isfinite(diff) else np.nan,
            }
        )

    out = pd.DataFrame(rows).sort_values(participant_col).reset_index(drop=True)

    # Optional quick sanity check:
    # print(out[[participant_col, "a_mean", "b_mean", diff_name]].describe())

    return out

