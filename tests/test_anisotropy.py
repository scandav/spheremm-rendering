import numpy as np
import pytest

import spheremm as sm
from spheremm.analysis import depolarization_index

ANISO = sm.Conductor("Al_Rakic", roughness=(0.03, 0.3))


def test_traces_are_concentric_around_the_image_centre():
    sphere = sm.Sphere()
    n = sm.polarization.normalize(np.random.default_rng(0).normal(size=(100, 3)) + [0, 0, 2])
    axis = np.array([0.0, 0, 1])  # towards the default camera
    t = sphere.trace_tangent(n, axis)
    np.testing.assert_allclose(np.sum(t * n, -1), 0, atol=1e-12)
    np.testing.assert_allclose(t[:, 2], 0, atol=1e-12)  # circles in planes ⟂ camera axis
    radial = n[:, :2] / np.linalg.norm(n[:, :2], axis=-1, keepdims=True)
    np.testing.assert_allclose(np.sum(t[:, :2] * radial, -1), 0, atol=1e-12)


def test_isotropic_roughness_tuple_equals_scalar():
    cam = sm.Camera(resolution=24)
    light = sm.DirectionalLight.from_camera(cam, 40, 20)
    a = sm.render_mueller(sm.Conductor("Cu", 0.2), cam, light).M
    b = sm.render_mueller(sm.Conductor("Cu", (0.2, 0.2)), cam, light).M
    np.testing.assert_allclose(a, b, atol=1e-12)


def test_anisotropy_needs_extended_source_to_change_polarisation():
    """Collimated light: one facet orientation per pixel -> normalised specular MM uniform.
    A finite-size source mixes facets -> spatially varying and depolarising."""
    cam = sm.Camera(resolution=48)
    kw = dict(phase_angle=40, azimuth=30)
    bright = lambda img: img[0][..., 0, 0] > 0.05 * img[0][..., 0, 0].max()

    point = sm.render_mueller(ANISO, cam, sm.DirectionalLight.from_camera(cam, **kw))
    m = point.normalized()[bright(point)]
    assert np.nanmax(np.std(m, axis=0)) < 1e-6

    wide = sm.render_mueller(ANISO, cam, sm.DirectionalLight.from_camera(cam, angular_radius=8, **kw))
    m = wide.normalized()[bright(wide)]
    assert np.nanmax(np.std(m, axis=0)) > 1e-3
    assert np.nanmin(depolarization_index(wide[0])[bright(wide)]) < 0.9995  # weak but present


def test_extended_source_conserves_irradiance():
    cam = sm.Camera(resolution=32)
    a = sm.render_mueller(sm.Lambertian(), cam, sm.DirectionalLight.from_camera(cam, 30))
    b = sm.render_mueller(sm.Lambertian(), cam, sm.DirectionalLight.from_camera(cam, 30, angular_radius=3))
    assert b[0][..., 0, 0].sum() == pytest.approx(a[0][..., 0, 0].sum(), rel=2e-3)


@pytest.mark.parametrize("azimuth", [0.0, 45.0, 90.0])  # mirrors that map pixels onto pixels
@pytest.mark.parametrize("light_kw", [{}, {"angular_radius": 5, "samples": 64}])
def test_mirror_symmetry_about_the_light_plane(azimuth, light_kw):
    """Concentric traces + light in a plane through the camera axis: the image is
    mirror-symmetric about that plane, with S2, S3 flipping sign (M -> P M P)."""
    cam = sm.Camera(resolution=48)
    light = sm.DirectionalLight.from_camera(cam, 30, azimuth, **light_kw)
    m = sm.render_mueller(ANISO, cam, light, supersample=1).normalized()
    # Rotate the image so the light plane is horizontal, mirror vertically, rotate back.
    a = np.radians(azimuth)

    def mirror(angle):  # Stokes mirror about a line at `angle` from the frame's x axis
        R = sm.polarization.rotator(angle)
        return R.T @ np.diag([1.0, 1, -1, -1]) @ R

    # The mirror plane contains the beam and the camera axis; its angle in the
    # light's (PSG) Stokes frame differs from the azimuth in the image.
    r, u, f = cam.basis
    k = light.direction
    x_in, y_in = sm.polarization.stokes_frame(k, u)
    v = -f - np.dot(-f, k) * k
    P_in, P_out = mirror(np.arctan2(v @ y_in, v @ x_in)), mirror(a)
    H = m.shape[0]
    c = (H - 1) / 2
    yy, xx = np.mgrid[0:H, 0:H]
    x, y = xx - c, c - yy  # image coordinates, y up
    ux, uy = np.cos(a), np.sin(a)
    dot_ = x * ux + y * uy
    xm, ym = 2 * dot_ * ux - x, 2 * dot_ * uy - y  # mirror across the light plane
    cm, rm = np.rint(xm + c).astype(int), np.rint(c - ym).astype(int)
    exact = (np.abs(xm + c - cm) < 1e-6) & (np.abs(c - ym - rm) < 1e-6)
    exact &= (cm >= 0) & (cm < H) & (rm >= 0) & (rm < H)
    asym = m[exact] - P_out @ m[rm[exact], cm[exact]] @ P_in
    assert exact.sum() > 50
    assert np.nanmax(np.abs(asym)) < 1e-9
