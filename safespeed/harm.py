"""Expected harm implied by the predicted speeds.

`harm_model` is the chance that the road user most at risk on the road (input
`crash_type`; a pedestrian when blank) is killed or seriously injured (MAIS3+F) in
a crash, averaged over the speed distribution the road invites. The distribution
is taken as normal: its median is the predicted v50 and its 85th percentile the
predicted v85. The injury-risk curves are Lubbe, Wu & Jeppsson (2022), Traffic
Safety Research 2:000006, doi:10.55329/vfma7555, for a 45-year-old
(weights/injury_curves.json).
"""
from __future__ import annotations

import numpy as np

# 20 evenly spaced quantiles of the speed distribution
_Q = np.array([0.025, 0.075, 0.125, 0.175, 0.225, 0.275, 0.325, 0.375, 0.425,
               0.475, 0.525, 0.575, 0.625, 0.675, 0.725, 0.775, 0.825, 0.875,
               0.925, 0.975])
_Z85 = 1.0364334      # standard-normal 85th percentile


def _normal_quantiles(p: np.ndarray) -> np.ndarray:
    """Inverse standard-normal CDF (Acklam's approximation, |error| < 1.15e-9)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    out = np.empty_like(p)
    lo, hi = p < 0.02425, p > 1 - 0.02425
    mid = ~(lo | hi)
    if lo.any():
        q = np.sqrt(-2 * np.log(p[lo]))
        out[lo] = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                  ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if hi.any():
        q = np.sqrt(-2 * np.log(1 - p[hi]))
        out[hi] = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                   ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p[mid] - 0.5
    r = q * q
    out[mid] = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    return out


_ZQ = _normal_quantiles(_Q)


def injury_risk(curves: dict, crash_type: str, travel_speed) -> np.ndarray:
    """P(killed or seriously injured) for a crash type at a travel speed (km/h)."""
    user = curves["crash_user"][crash_type]
    b = curves["coefficients"][user]
    v = np.asarray(travel_speed, "float64") * curves["closing_factor"][crash_type]
    z = b["b0"] + b["speed"] * v + b["age"] * curves["reference_age"]
    return 1.0 / (1.0 + np.exp(-z))


def harm_from_v50_v85(curves: dict, v50, v85, crash_types) -> np.ndarray:
    """Mean injury risk over a normal speed distribution pinned by v50 and v85."""
    v50 = np.asarray(v50, dtype="float64")
    v85 = np.asarray(v85, dtype="float64")
    sd = np.clip((v85 - v50) / _Z85, 0.5, None)
    grid = v50[:, None] + sd[:, None] * _ZQ[None, :]
    grid = np.clip(grid, 0.0, None)
    out = np.full(len(v50), np.nan)
    keys = np.asarray(crash_types, dtype=object)
    for k in sorted(set(keys)):
        m = keys == k
        out[m] = injury_risk(curves, k, grid[m]).mean(axis=1)
    return out
