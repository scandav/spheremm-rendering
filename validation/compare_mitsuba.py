"""Cross-validate spheremm against Mitsuba 3 (``uv sync --extra mitsuba``).

Mitsuba's S3 sign is flipped to spheremm's convention first (``Z``).

1. BSDF level, full 4×4: evaluate Mitsuba's polarised ``roughconductor`` and
   ``pplastic`` at random geometries and compare with spheremm's pBRDFs after
   re-expressing Mitsuba's implicit Stokes frames in spheremm's.
2. Image level, first column: render the sphere with Mitsuba's ``stokes``
   integrator (unpolarised light => Stokes image = M[:, 0]) and compare with
   spheremm's M[..., :, 0].  This checks camera, frames, and radiometry.

Run:  uv run python validation/compare_mitsuba.py
"""

import mitsuba as mi
import numpy as np

import spheremm as sm
from spheremm.materials import Geometry

mi.set_variant("llvm_ad_mono_polarized")

# Mitsuba's S3 has the opposite sign of spheremm's (right-circular = CCW looking into the beam).
Z = np.diag([1.0, 1, 1, -1])

ETA, K, ALPHA = 0.47, 2.4, 0.25  # a gold-like metal at ~550 nm
CONDUCTOR = (
    sm.Conductor(ior=complex(ETA, K), roughness=ALPHA),
    {"type": "roughconductor", "eta": ETA, "k": K, "alpha": ALPHA, "distribution": "ggx"},
)
ANISO = (
    sm.Conductor(ior=complex(ETA, K), roughness=(0.05, 0.4)),
    {"type": "roughconductor", "eta": ETA, "k": K, "alpha_u": 0.05, "alpha_v": 0.4, "distribution": "ggx"},
)
PLASTIC = (
    sm.Dielectric(ior=1.5, roughness=ALPHA, albedo=0.6),
    {"type": "pplastic", "int_ior": 1.5, "ext_ior": 1.0, "alpha": ALPHA,
     "diffuse_reflectance": {"type": "rgb", "value": 0.6}, "distribution": "ggx"},
)  # fmt: skip


def mitsuba_mueller(bsdf_dict, wi_cam, wo_light):
    """Mitsuba BSDF eval (includes cos θ_light); returns M and its implicit in/out x-axes."""
    bsdf = mi.load_dict(bsdf_dict)
    si = mi.SurfaceInteraction3f()
    v = lambda a: mi.Vector3f(*map(float, a))
    si.wi = v(wi_cam)
    M = bsdf.eval(mi.BSDFContext(), si, v(wo_light))
    M = Z @ np.array(M, dtype=float).reshape(4, 4) @ Z
    x_in = np.array(mi.mueller.stokes_basis(v(-wo_light))).ravel()
    x_out = np.array(mi.mueller.stokes_basis(v(wi_cam))).ravel()
    return M, x_in, x_out


def check_bsdf(name, material, bsdf_dict, n_trials=200, seed=0):
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(n_trials):
        wi, wo = (sm.polarization.normalize(rng.normal(size=3) * [1, 1, 0.3] + [0, 0, 1]) for _ in range(2))
        if wi[2] < 0.05 or wo[2] < 0.05:
            continue
        M_mi, x_in, x_out = mitsuba_mueller(bsdf_dict, wi, wo)
        k_in, k_out = -wo, wi
        g = Geometry(
            k_in, k_out, np.array([0.0, 0, 1]), x_in, np.cross(k_in, x_in), x_out, np.cross(k_out, x_out),
            t=np.array([1.0, 0, 0]),  # Mitsuba's alpha_u direction
        )
        M_sm = material.mueller_brdf(g, 550.0) * g.cos_i
        err = np.max(np.abs(M_sm - M_mi)) / max(M_mi[0, 0], 1e-9)
        worst = max(worst, err)
    print(f"[BSDF  ] {name:12s} max |ΔM|/M00 over random geometries: {worst:.2e}")
    return worst


def check_render(name, material, bsdf_dict, phase_angle=50.0, azimuth=30.0, res=64, spp=256):
    cam = sm.Camera(resolution=res, width=0.05)
    light = sm.DirectionalLight.from_camera(cam, phase_angle, azimuth)
    ours = sm.render_mueller(material, cam, light, supersample=4).M[0][..., :, 0]

    T = mi.ScalarTransform4f
    scene = mi.load_dict({
        "type": "scene",
        "integrator": {"type": "stokes", "integrator": {"type": "direct"}},
        "sensor": {
            "type": "orthographic",
            "to_world": T().look_at(origin=cam.origin.tolist(), target=cam.target.tolist(), up=cam.up.tolist())
            @ T().scale([cam.width / 2, cam.width / 2, 1]),
            "film": {"type": "hdrfilm", "width": res, "height": res, "rfilter": {"type": "box"}, "pixel_format": "luminance"},
            "sampler": {"type": "independent", "sample_count": spp},
        },
        "light": {"type": "directional", "direction": light.direction.tolist(), "irradiance": 1.0},
        "sphere": {"type": "sphere", "center": [0, 0, 0], "radius": 0.02, "bsdf": bsdf_dict},
    })  # fmt: skip
    img = np.array(mi.render(scene))
    theirs = img[..., [1, 4, 7, 10]] * np.diag(Z)  # channel 0 is the plain image, then S0..S3 as RGB triplets

    scale = np.max(ours[..., 0])
    diff = np.abs(ours - theirs) / scale
    mask = ours[..., 0] > 0.05 * scale
    print(f"[RENDER] {name:12s} mean |ΔS|/max(S0) = {diff[mask].mean():.2e}")
    for i, s in enumerate(["S0", "S1", "S2", "S3"]):
        print(f"           {s}: mean ours {ours[mask, i].mean():+.4f}  mitsuba {theirs[mask, i].mean():+.4f}")
    return ours, theirs


if __name__ == "__main__":
    cases = {"conductor": CONDUCTOR, "aniso_metal": ANISO, "pplastic": PLASTIC}
    for name, (mat, d) in cases.items():
        check_bsdf(name, mat, d)
    # Mitsuba's sphere has u = azimuth around its local z: with z facing the
    # camera, alpha_u runs along concentric circles centred in the image.
    for name, (mat, d) in cases.items():
        check_render(name, mat, d)
    # Negative control: swapping along/across must disagree (~7e-2), so the
    # agreement above really tests the trace orientation.
    swapped = sm.Conductor(ior=complex(ETA, K), roughness=(0.4, 0.05))
    check_render("aniso_swap", swapped, ANISO[1])
