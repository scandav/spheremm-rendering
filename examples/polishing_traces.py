"""Concentric polishing traces on a metal sphere: isotropic vs anisotropic,
collimated vs uncollimated (15° half-angle) source.  Figures go to output/traces_*.png.

Run:  uv run python examples/polishing_traces.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import spheremm as sm

out = Path(__file__).resolve().parents[1] / "output"
out.mkdir(exist_ok=True)

camera = sm.Camera(origin=[0, 0, 0.4], resolution=160, width=0.05)
sphere = sm.Sphere(radius=0.02)  # trace_axis=None: traces centred in the image
cases = {
    "isotropic": sm.library.get("brass", roughness=0.08),
    "traces": sm.library.get("brass", roughness=(0.03, 0.3)),
}
sources = {"collimated": {}, "15deg_source": {"angular_radius": 15.0, "samples": 1024}}

fig, axes = plt.subplots(len(sources), len(cases), figsize=(7, 7), constrained_layout=True)
for r, (src_name, light_kw) in enumerate(sources.items()):
    light = sm.DirectionalLight.from_camera(camera, phase_angle=30, azimuth=0, **light_kw)
    for c, (name, material) in enumerate(cases.items()):
        img = sm.render_mueller(material, camera, light, sphere, wavelengths=550)
        img.plot(0, title=f"brass {name}, {src_name}", output_file=out / f"traces_{name}_{src_name}_mm.png")
        m00 = img[0][..., 0, 0]
        axes[r, c].imshow(np.sqrt(m00 / m00.max()), cmap="gray")
        axes[r, c].set_title(f"{name}, {src_name}\n√M00", fontsize=9)
        axes[r, c].axis("off")
fig.savefig(out / "traces_m00.png", dpi=110)
print(f"saved {out}/traces_*.png")
