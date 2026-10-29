import numpy as np
import pytest

import spheremm as sm
from test_physics import coherency_eigs


@pytest.mark.parametrize("name", list(sm.library.MATERIALS))
def test_library_materials_render_physically(name):
    cam = sm.Camera(resolution=16)
    img = sm.render_mueller(
        sm.library.get(name), cam, sm.DirectionalLight.from_camera(cam, 40), wavelengths=[450, 550, 650]
    )
    M = img.M.reshape(-1, 4, 4)
    M = M[M[:, 0, 0] > 1e-6 * M[:, 0, 0].max()]
    assert len(M) > 0 and np.all(np.isfinite(M))
    assert min(coherency_eigs(m).min() for m in M) > -1e-9 * M[:, 0, 0].max()


def test_library_overrides():
    m = sm.library.get("brass", roughness=0.3)
    assert m.roughness == 0.3 and sm.library.MATERIALS["brass"].roughness != 0.3
    with pytest.raises(KeyError):
        sm.library.get("kryptonite")


def test_red_is_red():
    albedo = sm.library.get("red_abs").albedo
    assert albedo(650) > 5 * albedo(450)


@pytest.mark.parametrize("distance", [None, 0.5])
def test_phase_angle_sweep(distance):
    phases = [0, 25, 70, 120]
    images = sm.render_phase_sweep(
        sm.Lambertian(), phases, sm.Camera(resolution=8), light_distance=distance, azimuth=30
    )
    np.testing.assert_allclose([im.meta["phase_angle"] for im in images], phases, atol=1e-6)
