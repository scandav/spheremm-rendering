"""Render every library material at a few phase angles; figures go to output/.

Run:  uv run python examples/quickstart.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import spheremm as sm
from spheremm.plot import plot_intensity

out = Path(__file__).resolve().parents[1] / "output"
out.mkdir(exist_ok=True)

# 2 cm radius sphere, 40 cm away, imaged with a 5 cm field of view (telecentric).
camera = sm.Camera(origin=[0, 0, 0.4], resolution=200, width=0.05)
sphere = sm.Sphere(radius=0.02)
phase_angles = [0, 30, 60, 90]

for name in sm.library.MATERIALS:
    images = sm.render_phase_sweep(
        sm.library.get(name), phase_angles, camera, sphere, wavelengths=[450, 550, 650]
    )
    for img in images:
        phase = round(img.meta["phase_angle"])
        stem = out / f"{name}_phase{phase:03d}"
        img.save(f"{stem}.npz")
        img.plot(1, title=f"{name}  phase {phase}°  λ = 550 nm", output_file=f"{stem}_mm.png")
        fig = plot_intensity(img[1], f"{name} M00, phase {phase}°")
        fig.savefig(f"{stem}_m00.png", dpi=110)
        plt.close(fig)
    print(f"{name}: {len(images)} phase angles -> {out}/{name}_phase*.png")
