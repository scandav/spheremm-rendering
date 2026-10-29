"""Mueller-matrix analysis: normalisation, diattenuation, polarizance, depolarisation.

All functions take (..., 4, 4) arrays and return per-pixel quantities.
"""

from __future__ import annotations

import numpy as np


def normalize_mueller(M: np.ndarray, rel_threshold: float = 1e-6) -> np.ndarray:
    """``M / M00``; NaN where ``M00`` is below ``rel_threshold * max(M00)``."""
    m00 = M[..., 0, 0]
    valid = m00 > rel_threshold * np.max(m00)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = M / m00[..., None, None]
    return np.where(valid[..., None, None], m, np.nan)


def diattenuation(M: np.ndarray) -> np.ndarray:
    m = normalize_mueller(M)
    return np.linalg.norm(m[..., 0, 1:], axis=-1)


def polarizance(M: np.ndarray) -> np.ndarray:
    m = normalize_mueller(M)
    return np.linalg.norm(m[..., 1:, 0], axis=-1)


def depolarization_index(M: np.ndarray) -> np.ndarray:
    """Gil-Bernabeu depolarisation index: 1 = pure (non-depolarising), 0 = ideal depolariser."""
    m = normalize_mueller(M)
    return np.sqrt(np.maximum(np.sum(m**2, axis=(-2, -1)) - 1, 0) / 3)
