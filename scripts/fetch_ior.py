"""Download n,k data from the refractiveindex.info database (CC0) into src/spheremm/data/ior.

Run:  uv run --with pyyaml python scripts/fetch_ior.py
"""

import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import yaml

BASE = "https://raw.githubusercontent.com/polyanskiy/refractiveindex.info-database/main/database/data/"
OUT = Path(__file__).resolve().parents[1] / "src" / "spheremm" / "data" / "ior"

# output name -> database path
SOURCES = {
    "Al_Rakic": "main/Al/nk/Rakic.yml",
    "Ti_Johnson": "main/Ti/nk/Johnson.yml",
    "brass_Cu70Zn30": "other/alloys/Cu-Zn/nk/Querry-Cu70Zn30.yml",
    "steel_austenitic": "other/alloys/stainless steel/nk/Karlsson-austenitic.yml",
    "steel_ferritic": "other/alloys/stainless steel/nk/Karlsson-ferritic.yml",
    "Si3N4_Luke": "main/Si3N4/nk/Luke.yml",
}

WL_GRID_NM = np.arange(380.0, 781.0, 5.0)  # for formula-based entries


def formula_1(c, wl_um):
    """Sellmeier: n² - 1 = c0 + Σ c_i λ² / (λ² - c_{i+1}²)."""
    n2 = 1 + c[0]
    for b, l in zip(c[1::2], c[2::2]):
        n2 = n2 + b * wl_um**2 / (wl_um**2 - l**2)
    return np.sqrt(n2)


def convert(name: str, path: str) -> None:
    url = BASE + urllib.parse.quote(path)
    doc = yaml.safe_load(urllib.request.urlopen(url).read())
    entry = doc["DATA"][0]
    if entry["type"] == "tabulated nk":
        data = np.loadtxt(entry["data"].splitlines())
        wl, n, k = data[:, 0] * 1e3, data[:, 1], data[:, 2]
    elif entry["type"] == "formula 1":
        c = np.array(entry["coefficients"].split(), dtype=float)
        lo, hi = (float(x) * 1e3 for x in entry["wavelength_range"].split())
        wl = WL_GRID_NM[(WL_GRID_NM >= lo) & (WL_GRID_NM <= hi)]
        n, k = formula_1(c, wl / 1e3), np.zeros_like(wl)
    else:
        raise NotImplementedError(entry["type"])
    keep = (wl >= 300) & (wl <= 900)
    ref = " ".join(doc["REFERENCES"].split())
    header = f"{name}: refractiveindex.info {path} (CC0)\n{ref}\nwavelength [nm]"
    np.savetxt(OUT / f"{name}.eta.spd", np.c_[wl[keep], n[keep]], fmt="%.6g", header=header)
    np.savetxt(OUT / f"{name}.k.spd", np.c_[wl[keep], k[keep]], fmt="%.6g", header=header)
    print(f"{name:18s} {wl[keep].min():.0f}-{wl[keep].max():.0f} nm  <- {path}")


if __name__ == "__main__":
    for name, path in SOURCES.items():
        convert(name, path)
