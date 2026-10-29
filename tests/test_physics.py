import numpy as np
import pytest

import spheremm as sm
from spheremm.analysis import depolarization_index
from spheremm.fresnel import fresnel_amplitudes
from spheremm.polarization import jones_to_mueller, linear_polarizer, linear_retarder


def coherency_eigs(M):
    """Eigenvalues of the Cloude coherency matrix; all >= 0 iff M is physical."""
    from spheremm.polarization import _A, _A_INV

    # Inverse of M = A (J⊗J*) A^-1 generalised to a sum over pure components.
    K = _A_INV @ M @ _A
    H = K.reshape(2, 2, 2, 2).transpose(0, 2, 1, 3).reshape(4, 4)
    return np.linalg.eigvalsh((H + H.conj().T) / 2)


def test_jones_to_mueller_basics():
    np.testing.assert_allclose(jones_to_mueller(np.eye(2)), np.eye(4), atol=1e-12)
    np.testing.assert_allclose(jones_to_mueller(np.diag([1, 0])), linear_polarizer(0), atol=1e-12)
    # Quarter-wave plate at 45° turns horizontal into circular light.
    S = linear_retarder(np.pi / 2, np.pi / 4) @ [1, 1, 0, 0]
    np.testing.assert_allclose(np.abs(S), [1, 0, 0, 1], atol=1e-12)


def test_fresnel_limits():
    r_s, r_p = fresnel_amplitudes(np.array([1.0]), 1.5)
    np.testing.assert_allclose(r_s, -0.2)
    np.testing.assert_allclose(r_p, 0.2)  # Verdet convention
    brewster = np.cos(np.arctan(1.5))
    assert abs(fresnel_amplitudes(np.array([brewster]), 1.5)[1][0]) < 1e-12


@pytest.mark.parametrize("material", [sm.Conductor("Au", 0.2), sm.Dielectric(1.5, 0.2, albedo=0)])
def test_coaxial_center_is_mirror(material):
    """Back-scattering off the sphere's centre = normal-incidence mirror: diag(1, 1, -1, -1)."""
    img = sm.render_mueller(material, sm.Camera(resolution=33), supersample=1)
    m = img.normalized()[16, 16]
    np.testing.assert_allclose(m, np.diag([1, 1, -1, -1]), atol=1e-9)


@pytest.mark.parametrize(
    "material",
    [sm.Conductor("Cu", 0.3), sm.Dielectric("bk7", 0.3, albedo=0.7), sm.Lambertian(0.5)],
)
def test_rendered_matrices_are_physical(material):
    cam = sm.Camera(resolution=24)
    img = sm.render_mueller(material, cam, sm.DirectionalLight.from_camera(cam, 60, 20))
    M = img[0].reshape(-1, 4, 4)
    M = M[M[:, 0, 0] > 1e-6 * M[:, 0, 0].max()]
    eigs = np.array([coherency_eigs(m) for m in M])
    assert eigs.min() > -1e-9 * M[:, 0, 0].max()


def test_lambertian_is_ideal_depolarizer():
    img = sm.render_mueller(sm.Lambertian(0.8), sm.Camera(resolution=16))
    d = depolarization_index(img[0])
    assert np.nanmax(d) < 1e-12


def test_light_azimuth_zero_comes_from_the_right():
    cam = sm.Camera(resolution=32)
    img = sm.render_mueller(sm.Lambertian(), cam, sm.DirectionalLight.from_camera(cam, 60, 0))
    m00 = img[0][..., 0, 0]
    assert m00[:, 16:].sum() > 3 * m00[:, :16].sum()


def test_named_iors():
    assert sm.resolve_ior("bk7", 550) == pytest.approx(1.5046)
    gold = sm.resolve_ior("Au", 550)
    assert 0.3 < gold.real < 0.6 and 2.0 < gold.imag < 3.0
    with pytest.raises(KeyError):
        sm.resolve_ior("unobtainium", 550)


# --- Conventions (looking into the beam / camera image: x = right, y = up) ---


def test_right_circular_is_counterclockwise():
    from spheremm.polarization import jones_to_stokes

    for E in ([1, 1j], [1, -1j], [1, 0.5 + 0.3j]):
        E = np.array(E) / np.linalg.norm(E)
        t = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        field = np.real(E[None, :] * np.exp(-1j * t)[:, None])  # E(t), exp(-iωt)
        nxt = np.roll(field, -1, axis=0)
        ccw = np.mean(field[:, 0] * nxt[:, 1] - field[:, 1] * nxt[:, 0]) > 0
        S3 = jones_to_stokes(E)[3]
        assert (S3 > 0) == ccw


def test_linear_conventions_in_the_image():
    """Diffuse light leaves a sphere polarised radially, which pins S1/S2 to the image axes."""
    cam = sm.Camera(resolution=64)
    img = sm.render_mueller(sm.Dielectric(1.5, 0.3, albedo=1.0, specular=0.0), cam, supersample=1)
    m = img.normalized()
    c, edge = 32, 32 + int(0.95 * 0.02 / 0.025 * 32) - 1  # pixel near the limb
    d = c + int((edge - c) / np.sqrt(2))
    assert m[c, edge, 1, 0] > 0.05  # right limb: horizontal
    assert m[64 - edge, c, 1, 0] < -0.05  # top limb: vertical
    assert m[64 - d, d, 2, 0] > 0.05  # upper-right limb: +45° (CCW from horizontal)
    assert m[64 - d, 64 - d, 2, 0] < -0.05  # upper-left limb: -45°
