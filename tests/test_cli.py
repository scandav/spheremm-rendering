import numpy as np

from spheremm.cli import main


def test_cli_renders_all_parameters(tmp_path):
    main([
        "--material", "titanium", "--roughness", "0.03", "0.2", "--phase", "0", "45",
        "--azimuth", "90", "--angular-radius", "15", "--samples", "32", "--wavelengths", "450", "650",
        "--resolution", "16", "--supersample", "1", "--out", str(tmp_path), "--no-plot",
    ])  # fmt: skip
    files = sorted(tmp_path.glob("*.npz"))
    assert len(files) == 2
    data = np.load(files[1])
    assert data["M"].shape == (2, 16, 16, 4, 4)
    np.testing.assert_allclose(data["wavelengths"], [450, 650])
