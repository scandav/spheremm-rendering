"""Command line: ``spheremm-render --material brass --phase 0 30 60 --angular-radius 15``.

Writes one ``.npz`` (Mueller images, see :meth:`MuellerImage.save`) and, unless
``--no-plot``, a 4×4 figure and an M00 image per phase angle and wavelength.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import library
from .render import render_phase_sweep
from .scene import Camera, Sphere


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Render Mueller-matrix images of a sphere.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g = p.add_argument_group("material")
    g.add_argument("--material", default="brass", choices=list(library.MATERIALS))
    g.add_argument("--roughness", type=float, nargs="+", metavar="ALPHA",
                   help="GGX alpha, or two values: along / across the polishing traces (default: library value)")
    g.add_argument("--albedo", type=float, help="diffuse albedo override (dielectrics)")
    g = p.add_argument_group("illumination")
    g.add_argument("--phase", type=float, nargs="+", default=[30.0], help="phase angles (deg)")
    g.add_argument("--azimuth", type=float, default=0.0, help="light side in the image (deg, 0 = right, 90 = top)")
    g.add_argument("--angular-radius", type=float, default=0.0, help="source half-angle (deg); 0 = collimated")
    g.add_argument("--samples", type=int, help="source directions (default: automatic)")
    g.add_argument("--light-distance", type=float, help="point source at this distance (m) instead of a distant one")
    g = p.add_argument_group("camera / sphere")
    g.add_argument("--wavelengths", type=float, nargs="+", default=[550.0], help="nm")
    g.add_argument("--resolution", type=int, default=256)
    g.add_argument("--width", type=float, default=0.05, help="orthographic field of view (m)")
    g.add_argument("--fov", type=float, help="pinhole camera field of view (deg) instead of orthographic")
    g.add_argument("--camera-distance", type=float, default=0.4, help="m")
    g.add_argument("--radius", type=float, default=0.02, help="sphere radius (m)")
    g.add_argument("--supersample", type=int, default=3)
    g = p.add_argument_group("output")
    g.add_argument("--out", type=Path, default=Path("output"))
    g.add_argument("--no-plot", action="store_true")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    overrides = {}
    if args.roughness:
        if len(args.roughness) > 2:
            raise SystemExit("--roughness takes one or two values")
        overrides["roughness"] = args.roughness[0] if len(args.roughness) == 1 else tuple(args.roughness)
    if args.albedo is not None:
        overrides["albedo"] = args.albedo
    material = library.get(args.material, **overrides)

    camera = Camera(origin=[0, 0, args.camera_distance], resolution=args.resolution, width=args.width, fov=args.fov)
    light_kw = {}
    if args.light_distance is None:
        light_kw = dict(angular_radius=args.angular_radius, samples=args.samples)
    images = render_phase_sweep(
        material, args.phase, camera, Sphere(radius=args.radius), azimuth=args.azimuth,
        light_distance=args.light_distance, light_kw=light_kw,
        wavelengths=args.wavelengths, supersample=args.supersample,
    )  # fmt: skip

    args.out.mkdir(parents=True, exist_ok=True)
    for img in images:
        phase = img.meta["phase_angle"]
        stem = args.out / f"{args.material}_phase{phase:05.1f}_az{args.azimuth:g}_src{args.angular_radius:g}"
        img.save(f"{stem}.npz")
        print(f"{stem}.npz")
        if args.no_plot:
            continue
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from .plot import plot_intensity

        for i, wl in enumerate(img.wavelengths):
            title = f"{args.material}  phase {phase:.0f}°  λ = {wl:g} nm"
            img.plot(i, title=title, output_file=f"{stem}_{wl:g}nm_mm.png")
            fig = plot_intensity(img[i], title)
            fig.savefig(f"{stem}_{wl:g}nm_m00.png", dpi=110)
            plt.close(fig)


if __name__ == "__main__":
    main()
