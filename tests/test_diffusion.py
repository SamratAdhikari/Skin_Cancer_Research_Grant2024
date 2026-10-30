"""Unit tests for src/diffusion.py — skipped unless torch + diffusers are present.

These stay deliberately light: constructing the UNet + scheduler and running a
single forward / noising step is enough to catch API breakage without paying for
a full training or 1000-step sampling loop.
"""
import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

import diffusion as Dz  # noqa: E402


def test_set_seed_and_device():
    gen = Dz.set_seed(123)
    assert isinstance(gen, torch.Generator)
    dev = Dz.pick_device("cpu")
    assert dev.type == "cpu"


def test_lesion_dataset(tmp_path):
    paths = []
    for i in range(3):
        p = tmp_path / f"img_{i}.jpg"
        Image.fromarray(np.random.randint(0, 255, (40, 40, 3), "uint8")).save(p)
        paths.append(str(p))
    ds = Dz.LesionDataset(paths, image_size=32, augment=True)
    assert len(ds) == 3
    x = ds[0]
    assert tuple(x.shape) == (3, 32, 32)
    # DDPM expects inputs in [-1, 1].
    assert float(x.min()) >= -1.0001 and float(x.max()) <= 1.0001


def test_build_ddpm_forward_and_noise():
    unet, scheduler = Dz.build_ddpm(image_size=32, base_channels=16, norm_num_groups=32)
    x = torch.randn(1, 3, 32, 32)
    t = torch.tensor([10])
    out = unet(x, t).sample
    assert tuple(out.shape) == (1, 3, 32, 32)
    noisy = scheduler.add_noise(x, torch.randn_like(x), t)
    assert tuple(noisy.shape) == (1, 3, 32, 32)


def test_empty_dataset_raises():
    with pytest.raises(ValueError):
        Dz.LesionDataset([], image_size=32)
