"""Fresnel equations and the Mueller matrices of a single reflecting facet."""

from __future__ import annotations

import numpy as np

from .polarization import cross_normalized, dot, jones_to_mueller


def fresnel_amplitudes(cos_i: np.ndarray, eta: complex) -> tuple[np.ndarray, np.ndarray]:
    """Complex reflection amplitudes ``(r_s, r_p)`` from outside a medium of index ``eta``.

    ``eta = n + i k`` (``k >= 0`` absorbing, refractiveindex.info convention).
    ``r_p`` uses the Verdet sign convention (``r_p = -r_s`` at normal incidence),
    same as Mitsuba.
    """
    cos_i = np.clip(np.abs(cos_i), 0.0, 1.0).astype(complex)
    eta = complex(eta)
    sin2 = 1.0 - cos_i**2
    eta_cos_t = np.sqrt(eta**2 - sin2)  # principal root: Im >= 0, decaying wave
    r_s = (cos_i - eta_cos_t) / (cos_i + eta_cos_t)
    r_p = (eta**2 * cos_i - eta_cos_t) / (eta**2 * cos_i + eta_cos_t)
    return r_s, r_p


def fresnel_transmittance(cos_i: np.ndarray, eta: float) -> tuple[np.ndarray, np.ndarray]:
    """Power transmittances ``(T_s, T_p)`` across a dielectric interface (= 1 - R)."""
    r_s, r_p = fresnel_amplitudes(cos_i, eta)
    return 1.0 - np.abs(r_s) ** 2, 1.0 - np.abs(r_p) ** 2


def reflection_mueller(
    k_in: np.ndarray,
    k_out: np.ndarray,
    m: np.ndarray,
    eta: complex,
    x_in: np.ndarray,
    y_in: np.ndarray,
    x_out: np.ndarray,
    y_out: np.ndarray,
) -> np.ndarray:
    """Mueller matrix of specular reflection on a facet with normal ``m``.

    ``k_in`` / ``k_out`` are propagation directions (towards / away from the
    surface).  The result maps a Stokes vector in frame ``(x_in, y_in, k_in)``
    to one in frame ``(x_out, y_out, k_out)``.

    The reflection is built as a 3D Jones operator
    ``r_s s sᵀ + r_p p_out p_inᵀ`` (``p = k × s``) and projected on the given
    frames, so no explicit rotation angles (and their sign pitfalls) are needed.
    """
    cos_i = -dot(k_in, m)
    s = cross_normalized(k_in, m)
    p_in = np.cross(k_in, s)
    p_out = np.cross(k_out, s)
    r_s, r_p = fresnel_amplitudes(cos_i, eta)

    def proj(e_out, e_in):
        return r_s * dot(e_out, s) * dot(e_in, s) + r_p * dot(e_out, p_out) * dot(e_in, p_in)

    J = np.stack(
        [
            np.stack([proj(x_out, x_in), proj(x_out, y_in)], axis=-1),
            np.stack([proj(y_out, x_in), proj(y_out, y_in)], axis=-1),
        ],
        axis=-2,
    )
    return jones_to_mueller(J)


def transmission_stokes_row(
    k: np.ndarray, n: np.ndarray, eta: float, x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """First row/column of the Fresnel transmission Mueller matrix, in frame ``(x, y, k)``.

    Enough to describe light entering a depolarising body (row: how much of an
    incoming Stokes vector gets in) or leaving it (column: the Stokes vector of
    unpolarised light after refraction out).  ``k`` may point either way; only
    the plane of incidence matters.
    """
    cos_t = np.abs(dot(k, n))
    T_s, T_p = fresnel_transmittance(cos_t, eta)
    s = cross_normalized(k, n)
    c, sn = dot(s, x), dot(s, y)
    cos2, sin2 = c * c - sn * sn, 2 * c * sn
    half_sum, half_diff = 0.5 * (T_s + T_p), 0.5 * (T_s - T_p)
    return np.stack(
        [half_sum, half_diff * cos2, half_diff * sin2, np.zeros_like(half_sum)], axis=-1
    )
