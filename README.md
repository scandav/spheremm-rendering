# spheremm

Per-pixel **Mueller-matrix images of a sphere**, computed from its material:
complex refractive index (named, numeric, or tabulated vs. wavelength),
surface roughness and diffuse albedo.

```python
import spheremm as sm

camera = sm.Camera(origin=[0, 0, 0.4], resolution=256, width=0.05)  # orthographic, metres
light  = sm.DirectionalLight.from_camera(camera, phase_angle=45)    # collimated beam
img = sm.render_mueller(sm.library.get("brass"), camera, light,
                        sm.Sphere(radius=0.02), wavelengths=[450, 550, 650])

img.M            # (3, 256, 256, 4, 4) Mueller matrices (radiance per unit irradiance)
img.plot(1)      # normalised 4×4 figure at 550 nm (mm-plot, off-diagonal ±0.25)

# phase-angle sweep -> list of MuellerImage (img.meta["phase_angle"])
sweep = sm.render_phase_sweep(sm.library.get("pom"), [0, 30, 60, 90], camera)
```

## Setup

```bash
uv sync                    # numpy, matplotlib, mm-plot
uv sync --extra mitsuba    # + Mitsuba 3 (pip wheel, polarised variants included) for validation
uv run pytest              # physics/convention tests
uv run python examples/quickstart.py           # figures in output/
uv run python validation/compare_mitsuba.py    # cross-check against Mitsuba 3
```

Notebook: [notebooks/sphere_mm.ipynb](notebooks/sphere_mm.ipynb).

Command line (every parameter is an option, `--help` for the list):

```bash
uv run spheremm-render --material brass --roughness 0.03 0.3 \
    --phase 0 30 60 90 --azimuth 0 --angular-radius 15 --wavelengths 450 550 650
```

## How it works

A single sphere is convex, so light hits it once and goes to the camera: no
inter-reflections, so no Monte Carlo is needed. For every sub-pixel ray the
renderer:

1. intersects the sphere (normal `n`), finds the light direction `k_in` and
   view direction `k_out`;
2. evaluates the material's **polarimetric BRDF** `F(k_in, k_out, n)`, a 4×4
   Mueller matrix expressed directly from the light's (PSG) Stokes frame to the
   camera's (PSA) Stokes frame;
3. multiplies by `cos θ_i × irradiance` and averages over sub-pixels
   (`supersample`). The average is an incoherent sum, which gives the correct
   depolarisation where one pixel sees several normals, e.g. at the limb.

### Phase angle

The phase angle is the light–sphere–camera angle. `DirectionalLight.from_camera(camera, phase_angle, azimuth)`
and `PointLight.from_camera(camera, phase_angle, azimuth, distance)` place the
source; `azimuth` is the side the light comes from in the image (0° = right,
90° = top). `phase_angle=0` is coaxial back-scattering; beyond 90° only a
crescent is lit. The renderer ignores the light blocking the camera (or the
reverse), as with a beam splitter. `render_phase_sweep` loops over angles.

### Polishing traces (anisotropic finish) and uncollimated light

`roughness` can be a number (isotropic GGX α) or `(along, across)` the
polishing traces. The traces are concentric circles around
`Sphere.trace_axis`. By default that is the pole facing the camera, so the
circles are centred in the image (manual polishing). Library metals default
to `(0.03, 0.15)`.

```python
brass = sm.library.get("brass", roughness=(0.03, 0.3))
light = sm.DirectionalLight.from_camera(camera, phase_angle=30, angular_radius=15)
img = sm.render_mueller(brass, camera, light, sphere)
```

`angular_radius` is the **half-angle of the cone of light** reaching each point
(how uncollimated the source is, in degrees; 0 = collimated). The source is one
polarisation-state generator: its frame on the central ray is carried to every
direction in the cone by parallel transport, so mirror-symmetric setups render
mirror-symmetric images (tested). `samples=None` picks `max(64, 5 r²)`
directions. For α_along = 0.03 that gives max |Δm_ij| ≈ 0.005 at r = 5° and
≈ 0.003 with 1024 samples at r = 15°. Cost grows linearly with the sample count.

**Use uncollimated light with anisotropic finishes.** With a collimated beam and
an orthographic camera, each pixel reflects through a single facet orientation
(the half-vector). The traces then only change the intensity (the bow-tie
streak in M00); the normalised matrix stays uniform. A cone of directions makes
each pixel sum a range of facets. That produces spatially varying m_ij: with
a 15° source at 30° phase, brass with α = (0.03, 0.3), m12/m21 form a
four-lobe pattern of about ±0.1. Depolarisation stays weak (index ≥ 0.998), because nearby facets have nearly identical
Fresnel matrices. See `examples/polishing_traces.py`.

This is a rough *finish*: the surface geometry scatters differently along and
across the traces, but the metal's optical constants are isotropic. A
birefringent or dichroic material, with an n,k that depends on polarisation
direction, is a different effect and isn't modelled.

### Material library

`sm.library.get(name, **overrides)` ([library.py](src/spheremm/library.py)),
e.g. `sm.library.get("teflon", roughness=0.2, albedo=0.95)`:

| name | model | refractive index (source) | albedo (placeholder) |
|---|---|---|---|
| `brass` | Conductor, concentric traces | Cu70Zn30, Querry 1985 | – |
| `alu` | Conductor, concentric traces | Al, Rakić 1995 | – |
| `titanium` | Conductor, concentric traces | Ti, Johnson & Christy 1974 | – |
| `stainless_steel` | Conductor, concentric traces | austenitic (~316), Karlsson 1982 | – |
| `stainless_steel_ferritic` | Conductor, concentric traces | ferritic (~430/440), Karlsson 1982 | – |
| `silicon_nitride` | Dielectric | Si3N4, Luke 2015 (n ≈ 2.05) | 0.05 (dark grey) |
| `red_abs` | Dielectric | 1.54 | red edge at 600 nm, 0.04 → 0.55 |
| `red_bakelite` | Dielectric | 1.63 | red, 0.04 → 0.30 |
| `red_melamine` | Dielectric | 1.60 (literature 1.57–1.65) | red, 0.04 → 0.55 |
| `teflon` | Dielectric | 1.35 | 0.90 |
| `polypropylene` | Dielectric | 1.49 | 0.60 |
| `pom` | Dielectric | 1.48 | 0.80 |
| `peek` | Dielectric | 1.65 | beige, 0.25 → 0.55 |

Metal and Si3N4 spectra come from the refractiveindex.info database
(`uv run --with pyyaml python scripts/fetch_ior.py` regenerates them). The
polymers aren't in that database, so they use typical datasheet indices at 589 nm. Their
dispersion across the visible is about 1%, which has no visible effect on Fresnel. **Albedo and
roughness (default GGX α = 0.05) are placeholders**. They depend on the pigment
and the finish of your actual spheres; fit them to a measurement.

### Models

Material classes ([materials.py](src/spheremm/materials.py)):

| class | model | polarisation effects |
|---|---|---|
| `Conductor(ior, roughness)` | GGX microfacets (isotropic or anisotropic) + complex Fresnel | diattenuation **and** retardance |
| `Dielectric(ior, roughness, albedo, specular)` | GGX specular + diffuse lobe that is depolarised underneath and re-polarised by Fresnel transmission out (Baek et al. 2018) | specular diattenuation, diffuse polarisation, depolarisation |
| `Lambertian(albedo)` | ideal depolariser | none |

The Fresnel reflection is built as a 3D Jones operator and projected onto the
PSG/PSA frames, then converted to Mueller form. There are no rotation angles to
get wrong.

`ior` can be a number (`1.5`, `0.47 + 2.4j`, with `n + ik`, `k > 0` absorbing), a
name (`"Au"`, `"Ag"`, `"Al"`, `"Cu"`, `"Cr"`, `"bk7"`, `"water"`, … see
`sm.available_iors()`, data from Mitsuba 3), a function of wavelength in nm,
or `sm.ior_from_csv("Au.csv")` for a refractiveindex.info export.

Analysis ([analysis.py](src/spheremm/analysis.py)): normalised matrix,
diattenuation, polarizance, depolarisation index. The 4×4 figures use
[mm-plot](https://github.com/scandav/mm-plot) with its own colormap
(`sm.plot.plot_mueller`); `sm.plot.plot_intensity` shows the unnormalised M00.

## Conventions

All directions are trigonometric, **as seen looking into the beam**. For the
camera that means the image as displayed, with x = right and y = up.

| | meaning |
|---|---|
| `S1 > 0` | horizontal (image x) |
| `S2 > 0` | +45°, rotated counter-clockwise from horizontal |
| `S3 > 0` | right-circular = counter-clockwise rotation |
| PSG frame | `y` = camera "up" projected perpendicular to the beam (override with `light.up`) |
| time / index | fields `~exp(-iωt)`, absorbing index `n + ik` |

These are pinned by `tests/test_physics.py`. A mirror at normal incidence
(coaxial back-scatter, centre pixel) gives `diag(1, 1, -1, -1)`.

`Mitsuba 3` uses the opposite S3 sign: `M_mitsuba = Z M Z`, with `Z = diag(1, 1, 1, -1)`.

## Validation

`validation/compare_mitsuba.py` checks spheremm against Mitsuba 3.9
(`roughconductor` isotropic and anisotropic, `pplastic`):

* full 4×4 pBRDF at ~200 random geometries: max |ΔM|/M00 ≤ 1.6e-6 (float32);
* rendered Stokes images (`stokes` integrator, 64×64 px, 256 spp, phase 50°,
  unpolarised light = first column `M[:, 0]`): mean per-pixel |ΔS|/max(S0) of
  2.1e-4 (conductor), 7.0e-4 (anisotropic), 2.5e-4 (pplastic), over pixels with
  S0 > 5% of peak. The last line of the script is a negative control: swapping
  the two roughnesses gives 7.1e-2, so the trace orientation is really tested.
  Mitsuba's sphere parametrisation puts `alpha_u` along circles around its pole,
  i.e. the same concentric traces.

Last run: 26 Oct 2026, Mitsuba 3.9, `llvm_ad_mono_polarized`.

## Tests

`uv run pytest` (40 tests, ~3 s) pins:

* algebra and conventions (`test_physics.py`): Jones→Mueller, Fresnel limits,
  S1/S2/S3 signs in the image, light azimuth 0° = from the right, a mirror at
  normal incidence = `diag(1, 1, -1, -1)`, every rendered matrix physically
  realisable, Lambertian = ideal depolariser;
* anisotropy (`test_anisotropy.py`): traces concentric around the image centre,
  `(α, α)` = `α`, a collimated beam leaves the normalised matrix uniform while a
  cone source changes it, the cone conserves irradiance, and the image is
  mirror-symmetric about the light–camera plane;
* library and CLI (`test_library.py`, `test_cli.py`): every material renders
  physically, overrides work, red materials are red, phase sweeps with both
  light types, every CLI option.

## Output files

`MuellerImage.save(path)` writes a `.npz` with `M` (n_λ, H, W, 4, 4), `wavelengths` (nm)
and `coverage` (H, W). `M` is radiance per unit source irradiance (per unit
intensity / distance² for a `PointLight`), in the conventions below. Load it with
`np.load`.

`spheremm-render` writes to `--out` (default `output/`), one stem per phase angle,
`<material>_phase<φ>_az<azimuth>_src<angular radius>`:
`.npz`, plus per wavelength `_<λ>nm_mm.png` (normalised 4×4) and `_<λ>nm_m00.png`
(intensity). `--light-distance` switches to a `PointLight`, which is a true
point: `--angular-radius` and `--samples` are ignored.

## Limitations / next steps

* Sphere alone in the dark: no floor, walls, or environment light. Use Mitsuba
  if inter-reflections matter.
* With a collimated beam and an orthographic camera, every pixel has the same
  half-vector. The normalised *specular* matrix is then uniform across the
  sphere (correct for microfacet models). Spatial variation comes from the
  diffuse lobe, perspective, point lights and extended sources.
* Single-bounce microfacets only: no depolarisation from multiple scattering
  between facets, and no Rayleigh-Rice model for very smooth (σ ≪ λ) surfaces.
  The camera aperture isn't integrated; only the source size is.
* Smooth spheres (`roughness → 0`) give a glint smaller than a pixel. Use
  `roughness ≥ ~0.02` or increase `supersample`.
* Not modelled: thin films/coatings, birefringent or optically active bulk,
  polarised subsurface multiple scattering. Measured pBRDFs (e.g. the KAIST
  dataset) could plug in as another class with a `mueller_brdf` method.
