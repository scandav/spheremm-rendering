"""Per-pixel Mueller-matrix rendering of a sphere.

A single convex sphere has no inter-reflections, so the image is exactly the
single-scattering pBRDF evaluated at every visible point - no Monte Carlo
noise.  Pixels are anti-aliased by averaging Mueller matrices over a regular
sub-pixel grid (an incoherent sum, which is what a detector does); this is
also what produces the physical depolarisation at the sphere's limb.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import analysis
from .materials import Geometry
from .polarization import dot, normalize, stokes_frame
from .scene import Camera, DirectionalLight, PointLight, Sphere


@dataclass
class MuellerImage:
    """Result of :func:`render_mueller`.

    ``M``: (n_wavelengths, H, W, 4, 4) Mueller-matrix radiance for unit source
    irradiance (unpolarised input: ``M[..., :, 0]`` is the Stokes image).
    ``coverage``: fraction of each pixel covered by the sphere.
    """

    M: np.ndarray
    wavelengths: np.ndarray
    coverage: np.ndarray
    meta: dict = field(default_factory=dict)

    def __getitem__(self, i: int) -> np.ndarray:
        """Mueller image (H, W, 4, 4) at wavelength index ``i``."""
        return self.M[i]

    def normalized(self, i: int = 0) -> np.ndarray:
        """``M / M00`` (NaN where the sphere is dark or absent)."""
        return analysis.normalize_mueller(self.M[i])

    def plot(self, i: int = 0, **kw):
        from .plot import plot_mueller

        kw.setdefault("title", f"λ = {self.wavelengths[i]:g} nm")
        return plot_mueller(self.M[i], **kw)

    def save(self, path: str) -> None:
        np.savez_compressed(path, M=self.M, wavelengths=self.wavelengths, coverage=self.coverage)


def render_mueller(
    material,
    camera: Camera | None = None,
    light: DirectionalLight | PointLight | None = None,
    sphere: Sphere | None = None,
    wavelengths: float | list[float] = 550.0,
    supersample: int = 3,
    chunk: int = 1 << 18,
) -> MuellerImage:
    """Render the Mueller-matrix image of a sphere made of ``material``.

    ``camera`` defaults to an orthographic view along -z, ``light`` to a
    collimated beam coaxial with the camera (back-scattering).  Cost scales
    with ``supersample² × light samples × wavelengths``.
    """
    camera = camera or Camera()
    light = light or DirectionalLight.from_camera(camera, phase_angle=0.0)
    sphere = sphere or Sphere()
    wls = np.atleast_1d(np.asarray(wavelengths, dtype=float))

    o, d = camera.generate_rays(supersample)
    H, W, S = o.shape[:3]
    o, d = o.reshape(-1, 3), d.reshape(-1, 3)
    _, cam_up, cam_f = camera.basis

    out = np.zeros((len(wls), H * W * S, 4, 4))
    hit_all = np.zeros(H * W * S, dtype=bool)
    for start in range(0, len(o), chunk):
        sl = slice(start, start + chunk)
        hit, p, n = sphere.intersect(o[sl], d[sl])
        hit_all[sl] = hit
        idx = np.flatnonzero(hit)
        if idx.size == 0:
            continue
        p, n = p[idx], n[idx]
        k_out = -d[sl][idx]
        x_out, y_out = stokes_frame(k_out, cam_up)
        t = sphere.trace_tangent(n, -cam_f)
        for k_in, irradiance, (x_in, y_in) in light.illuminate(p, cam_up, -cam_f):
            g = Geometry(k_in, k_out, n, x_in, y_in, x_out, y_out, t)
            cos_i = g.cos_i
            ok = (cos_i > 0) & (g.cos_o > 0)
            scale = np.where(ok, cos_i * irradiance, 0.0)[:, None, None]
            for w, wl in enumerate(wls):
                out[w, start + idx] += np.nan_to_num(material.mueller_brdf(g, wl)) * scale

    M = out.reshape(len(wls), H, W, S, 4, 4).mean(axis=3)
    coverage = hit_all.reshape(H, W, S).mean(axis=-1)
    c = np.asarray(sphere.center, dtype=float)[None]
    k_in_c = normalize(sum(w.mean() * k[0] for k, w, _ in light.illuminate(c, cam_up, -cam_f)))
    k_out_c = -camera.generate_rays(1)[1][H // 2, W // 2, 0]
    phase = np.degrees(np.arccos(np.clip(-dot(k_in_c, k_out_c), -1, 1)))
    meta = dict(material=repr(material), supersample=supersample, phase_angle=float(phase))
    return MuellerImage(M=M, wavelengths=wls, coverage=coverage, meta=meta)


def render_phase_sweep(
    material,
    phase_angles,
    camera: Camera | None = None,
    sphere: Sphere | None = None,
    azimuth: float = 0.0,
    light_distance: float | None = None,
    light_kw: dict | None = None,
    **kw,
) -> list[MuellerImage]:
    """One :func:`render_mueller` per phase angle (degrees).

    The light rotates in the plane given by ``azimuth`` (0°: towards the image
    right).  Collimated beam by default; a point source at ``light_distance``
    metres from the camera target otherwise.  ``light_kw`` goes to the light
    (e.g. ``{"angular_radius": 2.0}``), ``kw`` to :func:`render_mueller`.
    """
    camera = camera or Camera()
    images = []
    for phase in np.atleast_1d(phase_angles):
        if light_distance is None:
            light = DirectionalLight.from_camera(camera, phase, azimuth, **(light_kw or {}))
        else:
            light = PointLight.from_camera(camera, phase, azimuth, distance=light_distance, **(light_kw or {}))
        images.append(render_mueller(material, camera, light, sphere, **kw))
    return images
