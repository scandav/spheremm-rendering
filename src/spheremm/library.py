"""Ready-made materials: ``spheremm.library.get("brass")``.

Metals and silicon nitride use measured n,k spectra (refractiveindex.info,
fetched by ``scripts/fetch_ior.py``).  The polymers are not in that database:
their IOR is a typical datasheet value at ~589 nm (dispersion over the visible
is ~1%, negligible for the Fresnel terms).

Metals have anisotropic roughness ``(along, across)`` their concentric polishing
traces (see ``Sphere.trace_axis``); everything else is isotropic.

The diffuse ``albedo`` (body colour) and ``roughness`` are *placeholders*:
they depend on pigment, filler and surface finish of your specific samples
and should be fitted to a measurement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .materials import Conductor, Dielectric

DEFAULT_ROUGHNESS = 0.05  # GGX alpha; ~polished
# Manually polished metals: concentric traces centred in the image, smooth
# along the traces and rougher across them (GGX alpha along, across).
METAL_ROUGHNESS = (0.03, 0.15)


@dataclass(frozen=True)
class EdgeSpectrum:
    """Smooth step from ``short`` (below ``edge`` nm) to ``long`` reflectance."""

    short: float
    long: float
    edge: float = 600.0
    width: float = 15.0

    def __call__(self, wavelength: float) -> float:
        return self.short + (self.long - self.short) / (1 + np.exp(-(wavelength - self.edge) / self.width))


def _red(long: float) -> EdgeSpectrum:
    return EdgeSpectrum(short=0.04, long=long, edge=600.0, width=15.0)


def _make():
    r, rm = DEFAULT_ROUGHNESS, METAL_ROUGHNESS
    return {
        # --- metals (measured n, k) ---
        "brass": Conductor("brass_Cu70Zn30", rm),  # Querry 1985, Cu70Zn30 (cartridge brass)
        "alu": Conductor("Al_Rakic", rm),  # Rakic 1995
        "titanium": Conductor("Ti_Johnson", rm),  # Johnson & Christy 1974
        "stainless_steel": Conductor("steel_austenitic", rm),  # Karlsson 1982, ~AISI 316
        "stainless_steel_ferritic": Conductor("steel_ferritic", rm),  # Karlsson 1982, ~AISI 430/440
        # --- ceramics (measured n) ---
        "silicon_nitride": Dielectric("Si3N4_Luke", r, albedo=0.05),  # bearing-grade balls are dark grey
        # --- polymers (datasheet n_D) ---
        "red_abs": Dielectric(1.54, r, albedo=_red(0.55)),
        "red_bakelite": Dielectric(1.63, r, albedo=_red(0.30)),  # phenol-formaldehyde, darker red
        "red_melamine": Dielectric(1.60, r, albedo=_red(0.55)),  # melamine-formaldehyde, n uncertain (1.57-1.65)
        "teflon": Dielectric(1.35, r, albedo=0.90),  # PTFE
        "polypropylene": Dielectric(1.49, r, albedo=0.60),  # natural, translucent
        "pom": Dielectric(1.48, r, albedo=0.80),  # acetal, natural white
        "peek": Dielectric(1.65, r, albedo=EdgeSpectrum(0.25, 0.55, edge=500.0, width=40.0)),  # natural beige
    }


MATERIALS = _make()


def get(name: str, **overrides):
    """Library material by name; keyword arguments override fields, e.g. ``roughness=0.2``."""
    from dataclasses import replace

    try:
        material = MATERIALS[name]
    except KeyError:
        raise KeyError(f"Unknown material {name!r}. Known: {', '.join(MATERIALS)}") from None
    return replace(material, **overrides)
