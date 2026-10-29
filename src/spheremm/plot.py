"""Figures for Mueller-matrix images (4×4 grid via mm-plot: github.com/scandav/mm-plot)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from mmplot import plot_mm

from .analysis import normalize_mueller

DIAG_LIMITS = (-1.0, 1.0)
OFF_DIAG_LIMITS = (-0.25, 0.25)


def plot_mueller(
    M: np.ndarray,
    title: str | None = None,
    diag_limits=DIAG_LIMITS,
    off_diag_limits=OFF_DIAG_LIMITS,
    **kw,
):
    """Normalised Mueller image ``M/M00`` (H, W, 4, 4) with ``mmplot.plot_mm``.

    Uses mm-plot's default colormap.  Extra keyword arguments (``output_file``,
    ``print_mean``, ``figsize``, ``cmap``...) go to ``plot_mm``.  Returns
    ``(fig, axes)``, or None if ``output_file`` is given.
    """
    m = np.moveaxis(normalize_mueller(M), (-2, -1), (0, 1))  # -> (4, 4, H, W)
    return plot_mm(m, title=title, diag_limits=diag_limits, off_diag_limits=off_diag_limits, **kw)


def plot_intensity(M: np.ndarray, title: str = "M00"):
    """Unnormalised M00 (radiance for unpolarised illumination)."""
    fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=True)
    im = ax.imshow(M[..., 0, 0], cmap="gray", vmin=0)
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, shrink=0.8)
    return fig
