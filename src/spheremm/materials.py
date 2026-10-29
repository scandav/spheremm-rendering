"""Polarimetric materials (pBRDFs).

Every material implements ``mueller_brdf(g, wavelength)`` which returns the
(..., 4, 4) Mueller-matrix BRDF ``F`` for the local geometry ``g``: the
reflected Stokes radiance is ``F @ S_in * cos(theta_i) * irradiance``.

Models
------
* :class:`Conductor` - rough metal: GGX microfacets + full complex Fresnel
  (diattenuation *and* retardance).  Same as Mitsuba's ``roughconductor``.
* :class:`Dielectric` - rough dielectric/plastic: polarising specular
  microfacet lobe + a diffuse lobe that is depolarised by subsurface
  scattering and re-polarised by refraction out of the surface
  (Baek et al. 2018, "Simultaneous acquisition of polarimetric SVBRDF and
  normals").  Same as Mitsuba's ``pplastic``.
* :class:`Lambertian` - ideal depolariser.

Refractive indices can be given as a number (``1.5``, ``0.2 + 3.4j``), as a
name (``"Au"``, ``"bk7"`` - see :func:`available_iors`), or as a function of
the wavelength in nm.  Use :func:`ior_from_csv` to load a refractiveindex.info
export.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Union

import numpy as np

from .fresnel import reflection_mueller, transmission_stokes_row
from .microfacet import ggx_D, smith_G1
from .polarization import cross_normalized, dot, normalize

IORLike = Union[float, complex, str, Callable[[float], complex]]
ValueLike = Union[float, Callable[[float], float]]
# GGX alpha: a number (isotropic) or (along, across) the polishing traces.
RoughnessLike = Union[float, tuple[float, float]]

_IOR_DIR = Path(__file__).parent / "data" / "ior"

# Constant IORs of common dielectrics (from Mitsuba 3, ~589 nm).
DIELECTRIC_IOR = {
    "vacuum": 1.0, "air": 1.000277, "water": 1.3330, "acetone": 1.36,
    "ethanol": 1.361, "carbon tetrachloride": 1.461, "glycerol": 1.4729,
    "benzene": 1.501, "silicone oil": 1.52045, "bromine": 1.661,
    "water ice": 1.31, "fused quartz": 1.458, "pyrex": 1.470,
    "acrylic glass": 1.49, "polypropylene": 1.49, "bk7": 1.5046,
    "sodium chloride": 1.544, "amber": 1.55, "pet": 1.5750, "diamond": 2.419,
}  # fmt: skip


def available_iors() -> list[str]:
    """Names accepted wherever an IOR is expected."""
    conductors = sorted({p.name.split(".")[0] for p in _IOR_DIR.glob("*.eta.spd")})
    return conductors + sorted(DIELECTRIC_IOR)


def _read_spd(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, comments="#")
    return data[:, 0], data[:, 1]


@lru_cache(maxsize=None)
def _tabulated(name: str) -> Callable[[float], complex]:
    wl_n, n = _read_spd(_IOR_DIR / f"{name}.eta.spd")
    wl_k, k = _read_spd(_IOR_DIR / f"{name}.k.spd")

    def ior(wavelength: float) -> complex:
        lo, hi = max(wl_n[0], wl_k[0]), min(wl_n[-1], wl_k[-1])
        if not lo <= wavelength <= hi:
            raise ValueError(f"{name}: data covers {lo:.0f}-{hi:.0f} nm, got {wavelength} nm")
        return complex(np.interp(wavelength, wl_n, n), np.interp(wavelength, wl_k, k))

    return ior


def ior_from_csv(path: str | Path, wavelength_unit: str = "um") -> Callable[[float], complex]:
    """Load ``wavelength, n[, k]`` columns (e.g. a refractiveindex.info CSV export)."""
    data = np.genfromtxt(path, delimiter=",", comments="#", skip_header=0)
    data = data[~np.isnan(data).any(axis=1)]
    scale = {"um": 1e3, "nm": 1.0}[wavelength_unit]
    wl = data[:, 0] * scale
    n = data[:, 1]
    k = data[:, 2] if data.shape[1] > 2 else np.zeros_like(n)
    return lambda w: complex(np.interp(w, wl, n), np.interp(w, wl, k))


def resolve_ior(ior: IORLike, wavelength: float) -> complex:
    if callable(ior):
        return complex(ior(wavelength))
    if isinstance(ior, str):
        if ior.lower() in DIELECTRIC_IOR:
            return complex(DIELECTRIC_IOR[ior.lower()])
        if (_IOR_DIR / f"{ior}.eta.spd").exists():
            return _tabulated(ior)(wavelength)
        raise KeyError(f"Unknown material {ior!r}. Known: {', '.join(available_iors())}")
    return complex(ior)


def _value(v: ValueLike, wavelength: float) -> float:
    return float(v(wavelength)) if callable(v) else float(v)


@dataclass
class Geometry:
    """Local scattering geometry, all arrays of shape (..., 3).

    ``k_in`` / ``k_out``: propagation directions of the incident / reflected
    light; ``n``: macroscopic surface normal; ``(x_in, y_in)`` and
    ``(x_out, y_out)``: Stokes reference frames of the light source and camera;
    ``t``: unit tangent along the polishing traces (anisotropic roughness only).
    """

    k_in: np.ndarray
    k_out: np.ndarray
    n: np.ndarray
    x_in: np.ndarray
    y_in: np.ndarray
    x_out: np.ndarray
    y_out: np.ndarray
    t: np.ndarray | None = None

    @property
    def cos_i(self) -> np.ndarray:
        return -dot(self.k_in, self.n)

    @property
    def cos_o(self) -> np.ndarray:
        return dot(self.k_out, self.n)


def _microfacet_specular(g: Geometry, eta: complex, roughness: RoughnessLike) -> np.ndarray:
    a_t, a_b = (roughness, roughness) if np.isscalar(roughness) else roughness
    if g.t is None:
        if a_t != a_b:
            raise ValueError("anisotropic roughness needs a trace direction (Geometry.t)")
        t = cross_normalized(g.n, g.n + 1.0)  # any tangent works when isotropic
    else:
        t = g.t
    wi, wo = -g.k_in, g.k_out  # both pointing away from the surface
    m = normalize(wi + wo)
    cos_i, cos_o = g.cos_i, g.cos_o
    D = ggx_D(m, g.n, t, a_t, a_b)
    G = smith_G1(wi, m, g.n, t, a_t, a_b) * smith_G1(wo, m, g.n, t, a_t, a_b)
    weight = D * G / (4.0 * np.maximum(cos_i * cos_o, 1e-12))
    F = reflection_mueller(g.k_in, g.k_out, m, eta, g.x_in, g.y_in, g.x_out, g.y_out)
    return F * weight[..., None, None]


@dataclass
class Conductor:
    """Rough metal.  ``ior``: complex index or name (e.g. ``"Au"``, ``"Al"``).

    ``roughness``: GGX alpha, or ``(along, across)`` the polishing traces
    (concentric around the camera axis by default, see ``Sphere.trace_axis``).
    """

    ior: IORLike = "Al"
    roughness: RoughnessLike = 0.1

    def mueller_brdf(self, g: Geometry, wavelength: float) -> np.ndarray:
        eta = resolve_ior(self.ior, wavelength)
        return _microfacet_specular(g, eta, self.roughness)


@dataclass
class Dielectric:
    """Rough dielectric with diffuse subsurface scattering (plastic, paint, skin...).

    ``albedo``: diffuse reflectance of the subsurface (0 for a clear,
    non-scattering dielectric).  ``specular``: scale of the specular lobe.
    """

    ior: IORLike = 1.5
    roughness: RoughnessLike = 0.1
    albedo: ValueLike = 0.5
    specular: float = 1.0

    def mueller_brdf(self, g: Geometry, wavelength: float) -> np.ndarray:
        eta = resolve_ior(self.ior, wavelength)
        if abs(eta.imag) > 0:
            raise ValueError("Dielectric needs a real IOR; use Conductor for absorbing media")
        eta = eta.real
        F = self.specular * _microfacet_specular(g, eta, self.roughness)

        albedo = _value(self.albedo, wavelength)
        if albedo > 0:
            t_in = transmission_stokes_row(g.k_in, g.n, eta, g.x_in, g.y_in)
            t_out = transmission_stokes_row(g.k_out, g.n, eta, g.x_out, g.y_out)
            F = F + (albedo / np.pi) * t_out[..., :, None] * t_in[..., None, :]
        return F


@dataclass
class Lambertian:
    """Perfectly depolarising diffuse surface."""

    albedo: ValueLike = 0.8

    def mueller_brdf(self, g: Geometry, wavelength: float) -> np.ndarray:
        F = np.zeros(g.n.shape[:-1] + (4, 4))
        F[..., 0, 0] = _value(self.albedo, wavelength) / np.pi
        return F
