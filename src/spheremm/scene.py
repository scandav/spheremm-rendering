"""Camera, light source and sphere geometry."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .polarization import cross_normalized, dot, normalize, stokes_frame


def _vec(v) -> np.ndarray:
    return np.asarray(v, dtype=float)


def _rotate(v: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation of ``v`` around unit ``axis`` by ``angle`` (rad)."""
    axis = normalize(axis)
    c, s = np.cos(angle), np.sin(angle)
    return v * c + np.cross(axis, v) * s + axis * dot(axis, v) * (1 - c)


@dataclass
class Sphere:
    """Sphere; ``trace_axis`` is the centre of concentric polishing traces.

    Traces are circles around ``trace_axis`` (a direction from the centre);
    ``None`` means the pole facing the camera, i.e. traces centred in the image.
    Only used by materials with anisotropic roughness.
    """

    center: np.ndarray = field(default_factory=lambda: np.zeros(3))
    radius: float = 0.02
    trace_axis: np.ndarray | None = None

    def trace_tangent(self, n: np.ndarray, default_axis: np.ndarray) -> np.ndarray:
        """Unit tangent along the concentric trace through each point (normal ``n``)."""
        axis = normalize(_vec(default_axis if self.trace_axis is None else self.trace_axis))
        return cross_normalized(np.broadcast_to(axis, n.shape), n)

    def intersect(self, o: np.ndarray, d: np.ndarray):
        """Nearest hit of rays ``o + t d`` (unit ``d``). Returns ``(hit, point, normal)``."""
        oc = o - _vec(self.center)
        b = dot(oc, d)
        c = dot(oc, oc) - self.radius**2
        disc = b * b - c
        hit = disc >= 0
        t = -b - np.sqrt(np.maximum(disc, 0.0))
        hit &= t > 0
        p = o + t[..., None] * d
        n = normalize(p - _vec(self.center))
        return hit, p, n


@dataclass
class Camera:
    """Pinhole (``fov`` in degrees) or orthographic (``fov=None``, ``width`` in m) camera.

    The output Stokes frame is the image frame: S1 > 0 = horizontal in the image,
    S2 > 0 = polarised along the image diagonal from bottom-left to top-right.
    """

    origin: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.4]))
    target: np.ndarray = field(default_factory=lambda: np.zeros(3))
    up: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 0.0]))
    resolution: int = 256
    fov: float | None = None
    width: float = 0.05

    @property
    def basis(self):
        f = normalize(_vec(self.target) - _vec(self.origin))
        r = normalize(np.cross(f, _vec(self.up)))
        u = np.cross(r, f)
        return r, u, f

    def generate_rays(self, supersample: int = 1):
        """Rays through a regular ``supersample²`` grid in every pixel.

        Returns origins and directions of shape (H, W, s², 3).
        """
        res, ss = self.resolution, supersample
        r, u, f = self.basis
        offs = (np.arange(ss) + 0.5) / ss
        px = (np.arange(res)[:, None] + offs[None, :]).reshape(-1)  # (res*ss,)
        sx = 2 * px / res - 1  # [-1, 1], left -> right
        sy = 1 - 2 * px / res  # [1, -1], top -> bottom
        X, Y = np.meshgrid(sx, sy)  # (res*ss, res*ss)
        X = X.reshape(res, ss, res, ss).transpose(0, 2, 1, 3).reshape(res, res, ss * ss)
        Y = Y.reshape(res, ss, res, ss).transpose(0, 2, 1, 3).reshape(res, res, ss * ss)
        X, Y = X[..., None], Y[..., None]
        if self.fov is None:
            o = _vec(self.origin) + X * (self.width / 2) * r + Y * (self.width / 2) * u
            d = np.broadcast_to(f, o.shape)
        else:
            t = np.tan(np.radians(self.fov) / 2)
            d = normalize(f + X * t * r + Y * t * u)
            o = np.broadcast_to(_vec(self.origin), d.shape)
        return o, d


def beam_direction(camera: Camera, phase_angle: float, azimuth: float = 0.0) -> np.ndarray:
    """Propagation direction of light arriving at the camera target at ``phase_angle`` (deg)."""
    r, u, f = camera.basis
    # Rotating around `axis` moves the source from the camera towards the
    # image direction at `azimuth` (counter-clockwise from the right).
    axis = _rotate(u, -f, np.radians(azimuth))
    return _rotate(f, axis, np.radians(phase_angle))


@dataclass
class DirectionalLight:
    """Distant source travelling along ``direction``; ``irradiance`` is the total.

    ``angular_radius`` (deg) > 0 makes it uncollimated: light arrives from a
    uniform cone of that half-angle (a lamp, diffuser or fibre output),
    integrated with ``samples`` deterministic directions.  ``samples=None``
    picks ``max(64, 5 r²)`` for a radius of r degrees, i.e. max |Δm_ij| ≈ 0.005
    for α_along = 0.03 traces; raise it for narrower lobes.  ``up`` fixes the generator (PSG) Stokes frame, ``y`` = ``up``
    projected perpendicular to the beam.  Defaults to the camera's up vector.
    """

    direction: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -1.0]))
    irradiance: float = 1.0
    up: np.ndarray | None = None
    angular_radius: float = 0.0
    samples: int | None = None

    def directions(self, mirror_ref: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Sample directions (N, 3) and weights (N,) covering the source's disk.

        The samples are mirror-symmetric about the plane through the beam and
        ``mirror_ref`` (the camera axis), so scenes that are symmetric about the
        light-camera plane stay exactly symmetric whatever the sample count.
        """
        d = normalize(_vec(self.direction))
        if self.angular_radius <= 0:
            return d[None], np.ones(1)
        e1 = None if mirror_ref is None else _vec(mirror_ref) - dot(_vec(mirror_ref), d) * d
        if e1 is None or np.linalg.norm(e1) < 1e-9:
            e1 = cross_normalized(d, d + 1.0)
        e1 = normalize(e1)
        e2 = np.cross(d, e1)
        # Half a Vogel spiral (equal solid angle per sample) + its mirror image.
        samples = self.samples or max(64, int(np.ceil(5 * self.angular_radius**2)))
        half = max(samples // 2, 1)
        i = np.arange(half) + 0.5
        cos_max = np.cos(np.radians(self.angular_radius))
        cos_t = np.tile(1 - i / half * (1 - cos_max), 2)
        sin_t = np.sqrt(1 - cos_t**2)
        phi = i * np.pi * (3 - np.sqrt(5))
        phi = np.concatenate([phi, -phi])
        dirs = cos_t[:, None] * d + sin_t[:, None] * (np.cos(phi)[:, None] * e1 + np.sin(phi)[:, None] * e2)
        return dirs, np.full(2 * half, 1.0 / (2 * half))

    @classmethod
    def from_camera(cls, camera: Camera, phase_angle: float = 0.0, azimuth: float = 0.0, **kw):
        """Beam at ``phase_angle`` degrees (light-object-camera angle) from the camera axis.

        ``azimuth`` is measured in the image plane from the image's right
        (0°: light comes from the right of the camera, 90°: from above).
        ``phase_angle=0`` is coaxial back-scattering.
        """
        return cls(direction=beam_direction(camera, phase_angle, azimuth), **kw)

    def illuminate(self, p: np.ndarray, default_up: np.ndarray, to_camera: np.ndarray | None = None):
        """List of ``(k_in, irradiance, (x_in, y_in))``, one per source sample."""
        up = default_up if self.up is None else _vec(self.up)
        # One PSG for the whole source: its frame on the central ray is carried
        # to every sample direction by the minimal rotation (parallel transport).
        d0 = normalize(_vec(self.direction))
        x0, _ = stokes_frame(d0, up)
        out = []
        for d, w in zip(*self.directions(to_camera)):
            axis = np.cross(d0, d)
            s = np.linalg.norm(axis)
            x = x0 if s < 1e-12 else _rotate(x0, axis, np.arctan2(s, dot(d0, d)))
            k, x = np.broadcast_to(d, p.shape), np.broadcast_to(x, p.shape)
            out.append((k, np.full(p.shape[:-1], w * self.irradiance), (x, np.cross(k, x))))
        return out


@dataclass
class PointLight:
    """Isotropic point source with radiant ``intensity`` (W/sr)."""

    position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.5]))
    intensity: float = 1.0
    up: np.ndarray | None = None

    @classmethod
    def from_camera(
        cls, camera: Camera, phase_angle: float = 0.0, azimuth: float = 0.0, distance: float = 0.5, **kw
    ):
        """Source ``distance`` m from the camera target, at ``phase_angle`` (see DirectionalLight)."""
        position = _vec(camera.target) - distance * beam_direction(camera, phase_angle, azimuth)
        return cls(position=position, **kw)

    def illuminate(self, p: np.ndarray, default_up: np.ndarray, to_camera: np.ndarray | None = None):
        v = p - _vec(self.position)
        dist2 = dot(v, v)
        k = normalize(v)
        up = default_up if self.up is None else _vec(self.up)
        return [(k, self.intensity / dist2, stokes_frame(k, up))]
