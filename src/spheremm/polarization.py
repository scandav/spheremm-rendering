"""Stokes / Jones / Mueller algebra.

Conventions (used everywhere in this package)
---------------------------------------------
* A Stokes vector is always expressed in a *reference frame* ``(x, y, k)``:
  ``k`` is the propagation direction and ``x, y`` are unit vectors
  perpendicular to it with ``x × y = k`` (right-handed).
All angles and handedness are "trigonometric" as seen *looking into the beam*
(for the camera: in the image, x = right, y = up):

* ``S1 > 0``: horizontal (along x)
* ``S2 > 0``: +45°, i.e. x rotated 45° counter-clockwise (towards y)
* ``S3 > 0``: right-circular = E rotating counter-clockwise

With fields ``~ exp(-iωt)`` (absorbing index ``n + ik``) this is
``S3 = -2 Im(Ex Ey*)``.  Mitsuba 3 uses the opposite S3 sign.

All functions are vectorised over arbitrary leading dimensions ``(...)``.
"""

from __future__ import annotations

import numpy as np

# Maps the coherency vector [ExEx*, ExEy*, EyEx*, EyEy*] to [S0, S1, S2, S3].
# S3 = -2 Im(Ex Ey*): with fields ~ exp(-iωt) (and hence absorbing n + ik), S3 > 0
# means E rotates counter-clockwise when looking into the beam (= in the camera
# image).  Mitsuba 3 uses the opposite S3 sign: M_mitsuba = Z M Z, Z = diag(1,1,1,-1)
# (validation/compare_mitsuba.py).
_A = np.array(
    [
        [1, 0, 0, 1],
        [1, 0, 0, -1],
        [0, 1, 1, 0],
        [0, 1j, -1j, 0],
    ],
    dtype=complex,
)
_A_INV = np.linalg.inv(_A)


def normalize(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(a * b, axis=-1)


def cross_normalized(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``normalize(a × b)``, with an arbitrary perpendicular vector where ``a ∥ b``.

    Used for the s-axis of a plane of incidence, which is undefined at normal
    incidence (where s- and p-polarisation are equivalent anyway).
    """
    c = np.cross(a, b)
    norm = np.linalg.norm(c, axis=-1, keepdims=True)
    bad = norm[..., 0] < 1e-9
    if np.any(bad):
        alt = np.where(np.abs(a[..., :1]) < 0.9, [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
        c = np.where(bad[..., None], np.cross(a, alt), c)
        norm = np.linalg.norm(c, axis=-1, keepdims=True)
    return c / norm


def stokes_frame(k: np.ndarray, up: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Right-handed Stokes frame ``(x, y)`` for propagation direction ``k``.

    ``y`` is the projection of ``up`` on the plane perpendicular to ``k`` and
    ``x = y × k``.  For light travelling towards a camera whose image "up" is
    ``up``, ``x`` is the image's "right" direction, i.e. S1 > 0 means
    horizontally polarised in the picture.
    """
    k = normalize(k)
    up = np.broadcast_to(up, k.shape)
    y = up - dot(up, k)[..., None] * k
    # Fallback if up is parallel to k: use any perpendicular vector.
    bad = np.linalg.norm(y, axis=-1) < 1e-9
    if np.any(bad):
        alt = np.where(np.abs(k[..., :1]) < 0.9, [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
        y_alt = alt - dot(alt, k)[..., None] * k
        y = np.where(bad[..., None], y_alt, y)
    y = normalize(y)
    x = np.cross(y, k)
    return x, y


def jones_to_mueller(J: np.ndarray) -> np.ndarray:
    """Convert (..., 2, 2) complex Jones matrices to (..., 4, 4) real Mueller matrices."""
    K = np.einsum("...ij,...kl->...ikjl", J, J.conj())
    K = K.reshape(*J.shape[:-2], 4, 4)
    return np.real(_A @ K @ _A_INV)


def jones_to_stokes(E: np.ndarray) -> np.ndarray:
    """Stokes vector(s) (..., 4) of Jones vector(s) (..., 2) (fields ``~ exp(-iωt)``)."""
    K = np.einsum("...i,...k->...ik", E, E.conj()).reshape(*E.shape[:-1], 4)
    return np.real(K @ _A.T)


def rotator(theta: np.ndarray | float) -> np.ndarray:
    """Mueller matrix re-expressing a Stokes vector in a frame rotated by ``theta`` (rad)."""
    c, s = np.cos(2 * np.asarray(theta)), np.sin(2 * np.asarray(theta))
    M = np.zeros(np.shape(theta) + (4, 4))
    M[..., 0, 0] = M[..., 3, 3] = 1
    M[..., 1, 1] = M[..., 2, 2] = c
    M[..., 1, 2] = s
    M[..., 2, 1] = -s
    return M


def linear_polarizer(theta: float = 0.0) -> np.ndarray:
    """Ideal linear polariser with transmission axis at angle ``theta`` (rad) from x."""
    c, s = np.cos(2 * theta), np.sin(2 * theta)
    return 0.5 * np.array(
        [[1, c, s, 0], [c, c * c, c * s, 0], [s, c * s, s * s, 0], [0, 0, 0, 0]]
    )


def linear_retarder(delta: float, theta: float = 0.0) -> np.ndarray:
    """Linear retarder with retardance ``delta`` and fast axis at ``theta`` (rad)."""
    J = np.array([[1, 0], [0, np.exp(1j * delta)]])
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return jones_to_mueller(R @ J @ R.T)


def stokes_to_dolp_aolp(S: np.ndarray, eps: float = 1e-12):
    """Degree and angle (rad, in [0, π)) of linear polarisation of (..., 4) Stokes vectors."""
    dolp = np.hypot(S[..., 1], S[..., 2]) / np.maximum(S[..., 0], eps)
    aolp = np.mod(0.5 * np.arctan2(S[..., 2], S[..., 1]), np.pi)
    return dolp, aolp
