from __future__ import annotations
import numpy as np
import pandas as pd


def apply_filters(
    df: pd.DataFrame,
    *,
    filters: dict | None,
) -> pd.DataFrame:
    """
    Apply filtering rules specified in the config.

    All failed filters invalidate the entire row (set to NaN),
    preserving row structure for downstream analysis.
    """
    if not filters:
        return df

    out = df.copy()

    # Start with a mask of rows to invalidate
    invalidate = pd.Series(False, index=out.index)

    # --- absolute error filter ---
    if "max_abs_error_deg" in filters:
        max_err = filters["max_abs_error_deg"]
        invalidate |= out["error"].abs() > max_err

    # --- movement speed filter ---
    if "mean_velocity" in filters:
        speed_cfg = filters["mean_velocity"]

        if "lower" in speed_cfg:
            invalidate |= out["mean_velocity"] < speed_cfg["lower"]

        if "upper" in speed_cfg:
            invalidate |= out["mean_velocity"] > speed_cfg["upper"]

    # --- conditional invalidation rules ---
    for rule in filters.get("invalidate_conditions", []):
        when = rule["when"]

        mask = pd.Series(True, index=out.index)
        for col, val in when.items():
            mask &= out[col] == val

        invalidate |= mask

    ID_COLS = ["participant", "block", "trial_num"]
    VALUE_COLS = [c for c in df.columns if c not in ID_COLS]
    # --- apply invalidation ---
    out["invalid"] = invalidate
    out.loc[invalidate, VALUE_COLS] = pd.NA

    return out