"""GGX microfacet distribution (isotropic or anisotropic) with separable Smith shadowing.

Identical to Mitsuba's ``ggx`` distribution: ``alpha_t`` is the roughness along
the tangent ``t`` (Mitsuba's ``alpha_u``), ``alpha_b`` along ``b = n × t``.
"""

from __future__ import annotations

import numpy as np

from .polarization import dot


def ggx_D(m: np.ndarray, n: np.ndarray, t: np.ndarray, alpha_t: float, alpha_b: float) -> np.ndarray:
    b = np.cross(n, t)
    mt, mb, mn = dot(m, t) / alpha_t, dot(m, b) / alpha_b, dot(m, n)
    denom = mt * mt + mb * mb + mn * mn
    return np.where(mn > 0, 1.0 / (np.pi * alpha_t * alpha_b * denom * denom), 0.0)


def smith_G1(
    v: np.ndarray, m: np.ndarray, n: np.ndarray, t: np.ndarray, alpha_t: float, alpha_b: float
) -> np.ndarray:
    b = np.cross(n, t)
    cos_v = dot(v, n)
    xy_alpha2 = (alpha_t * dot(v, t)) ** 2 + (alpha_b * dot(v, b)) ** 2
    tan2_alpha = xy_alpha2 / np.maximum(cos_v * cos_v, 1e-30)
    g = 2.0 / (1.0 + np.sqrt(1.0 + tan2_alpha))
    return np.where(dot(v, m) * cos_v > 0, g, 0.0)
