"""
diffusion.py
------------
A compact DDPM (denoising diffusion probabilistic model) for **synthetic
augmentation of rare skin-lesion classes**.

Motivation
==========
HAM10000 is heavily imbalanced. Even within the malignant group, ``akiec``
(actinic keratosis / intraepithelial carcinoma, ~327 images) is far rarer than
``mel`` or ``bcc``. Class weights help, but the model still sees very few
distinct rare-class examples. We train a small unconditional DDPM *per rare
class* and synthesize extra images, then run an **ablation**: retrain the
malignant sub-classifier with vs. without the synthetic data and report the
change in that class's recall / macro-F1.

Design choices (time-boxed for a free Colab T4)
===============================================
* 64x64 generation (a UNet at 224x224 is far too slow to train in the budget).
  Synthetic images are bicubically upscaled to 224 for the classifier — an
  honest, disclosed limitation.
* One *unconditional* model per class (train it only on that class's images),
  which is simpler and more stable on a few hundred images than a single
  class-conditional model.
* Optional lightweight EMA of the weights (markedly better samples for almost
  no cost).

This module is intentionally decoupled from the TensorFlow classifier stack:
it only ever *reads* lesion JPEGs and *writes* synthetic PNGs. The two halves
communicate through the filesystem, so the mixed TF+PyTorch stack never has to
coexist in one process.

Heavy imports (``torch`` at module load; ``diffusers`` lazily inside the
functions that need it) are deferred where possible so the module can be
imported for its CLI/help without diffusers installed.
"""

from __future__ import annotations

import argparse
import glob
import math
import os
from typing import Sequence

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

DEFAULT_IMAGE_SIZE = 64
DEFAULT_SEED = 2024


# --------------------------------------------------------------------- utilities
def _require_diffusers():
    """Import diffusers with a helpful error if it is not installed."""
    try:
        import diffusers  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "This function needs the 'diffusers' library.\n"
            "    pip install diffusers accelerate\n"
            "(On Colab it is usually pre-installed or installs in seconds.)"
        ) from exc
    return diffusers


def set_seed(seed: int = DEFAULT_SEED) -> torch.Generator:
    """Seed Python/torch RNGs and return a seeded ``torch.Generator``."""
    import random

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    return gen


def pick_device(device: str | None = None) -> torch.device:
    """Resolve the compute device (CUDA if available, else CPU)."""
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------- dataset
class LesionDataset(Dataset):
    """A tiny image dataset that maps a list of JPEG paths to model tensors.

    Images are resized to ``image_size`` and scaled to ``[-1, 1]`` (the range
    a DDPM expects). Optional horizontal/vertical flips add cheap variety —
    dermatoscopic lesions have no canonical orientation, so this is safe.
    """

    def __init__(
        self,
        image_paths: Sequence[str],
        image_size: int = DEFAULT_IMAGE_SIZE,
        augment: bool = True,
    ) -> None:
        self.image_paths = list(image_paths)
        if not self.image_paths:
            raise ValueError("LesionDataset received an empty image list.")
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.image_paths[idx]).convert("RGB").resize(
            (self.image_size, self.image_size), Image.BICUBIC
        )
        arr = torch.from_numpy(_to_float_chw(img))  # (3,H,W) in [0,1]
        if self.augment:
            if torch.rand(1).item() < 0.5:
                arr = torch.flip(arr, dims=[2])  # horizontal
            if torch.rand(1).item() < 0.5:
                arr = torch.flip(arr, dims=[1])  # vertical
        return arr * 2.0 - 1.0  # -> [-1, 1]


def _to_float_chw(img: Image.Image):
    """PIL RGB -> float32 CHW array in [0,1] (numpy, no torchvision needed)."""
    import numpy as np

    arr = np.asarray(img, dtype="float32") / 255.0  # (H,W,3)
    return arr.transpose(2, 0, 1).copy()  # (3,H,W)


# ------------------------------------------------------------------------- model
def build_ddpm(
    image_size: int = DEFAULT_IMAGE_SIZE,
    num_train_timesteps: int = 1000,
    base_channels: int = 128,
    norm_num_groups: int = 32,
):
    """Construct a ``UNet2DModel`` + ``DDPMScheduler`` sized for small images.

    The block configuration is the standard diffusers 64px recipe (with one
    self-attention stage), optionally scaled down via ``base_channels`` to fit
    a tighter time budget. ``norm_num_groups`` is clamped to ``base_channels``
    so smaller widths (which must stay divisible by the group count) still
    build — the default width of 128 is unaffected.
    """
    diffusers = _require_diffusers()
    from diffusers import DDPMScheduler, UNet2DModel

    c = base_channels
    groups = min(norm_num_groups, c)
    unet = UNet2DModel(
        sample_size=image_size,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        norm_num_groups=groups,
        block_out_channels=(c, c, c * 2, c * 2, c * 4, c * 4),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )
    scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps)
    return unet, scheduler


class _EMA:
    """Minimal exponential-moving-average of model parameters.

    Version-agnostic (doesn't depend on ``diffusers.training_utils.EMAModel``),
    which keeps the notebook robust across diffusers releases.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = {
            k: p.detach().clone() for k, p in model.named_parameters() if p.requires_grad
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for k, p in model.named_parameters():
            if p.requires_grad and k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: torch.nn.Module) -> None:
        for k, p in model.named_parameters():
            if k in self.shadow:
                p.data.copy_(self.shadow[k])


# ------------------------------------------------------------------------- train
def train_ddpm(
    image_paths: Sequence[str],
    out_dir: str,
    image_size: int = DEFAULT_IMAGE_SIZE,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-4,
    num_train_timesteps: int = 1000,
    base_channels: int = 128,
    use_ema: bool = True,
    device: str | None = None,
    seed: int = DEFAULT_SEED,
    log_every: int = 10,
    max_minutes: float | None = None,
) -> str:
    """Train an unconditional DDPM on a single class's images and save it.

    Args:
        image_paths: JPEG paths for ONE class (e.g. all ``akiec`` images).
        out_dir: directory to write the trained UNet + scheduler config to.
        epochs / batch_size / lr / num_train_timesteps: training hyperparams.
        base_channels: UNet width (lower = faster, lower quality).
        use_ema: keep an EMA copy of the weights and save that (better samples).
        device: 'cuda' / 'cpu' (auto if None).
        max_minutes: optional wall-clock cap for a time-boxed Colab run; if the
            elapsed training time exceeds this after an epoch, stop early.

    Returns:
        ``out_dir`` (containing ``unet/`` and ``scheduler/`` subfolders usable
        by :func:`load_ddpm` / :func:`generate_images`).
    """
    import time

    diffusers = _require_diffusers()
    dev = pick_device(device)
    set_seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    dataset = LesionDataset(image_paths, image_size=image_size, augment=True)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=len(dataset) > batch_size
    )

    unet, scheduler = build_ddpm(image_size, num_train_timesteps, base_channels)
    unet.to(dev)
    optimizer = torch.optim.AdamW(unet.parameters(), lr=lr)
    ema = _EMA(unet) if use_ema else None

    print(
        f"[diffusion] training on {len(dataset)} images @ {image_size}px, "
        f"{epochs} epochs, batch {batch_size}, device={dev}"
    )
    start = time.time()
    global_step = 0
    for epoch in range(epochs):
        unet.train()
        running = 0.0
        for clean in loader:
            clean = clean.to(dev)
            noise = torch.randn_like(clean)
            bsz = clean.shape[0]
            timesteps = torch.randint(
                0, scheduler.config.num_train_timesteps, (bsz,), device=dev
            ).long()
            noisy = scheduler.add_noise(clean, noise, timesteps)

            noise_pred = unet(noisy, timesteps).sample
            loss = F.mse_loss(noise_pred, noise)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            if ema is not None:
                ema.update(unet)

            running += loss.item()
            global_step += 1

        if epoch % log_every == 0 or epoch == epochs - 1:
            elapsed = (time.time() - start) / 60.0
            print(
                f"  epoch {epoch:4d}/{epochs}  loss={running / len(loader):.4f}  "
                f"({elapsed:.1f} min)"
            )

        if max_minutes is not None and (time.time() - start) / 60.0 >= max_minutes:
            print(f"[diffusion] hit time budget ({max_minutes} min) at epoch {epoch}; stopping.")
            break

    # Save (EMA weights if we kept them).
    if ema is not None:
        ema.copy_to(unet)
    unet.save_pretrained(os.path.join(out_dir, "unet"))
    scheduler.save_pretrained(os.path.join(out_dir, "scheduler"))
    print(f"[diffusion] saved model to {out_dir}")
    return out_dir


# -------------------------------------------------------------------- load / gen
def load_ddpm(model_dir: str, device: str | None = None):
    """Load a UNet + scheduler saved by :func:`train_ddpm`."""
    _require_diffusers()
    from diffusers import DDPMScheduler, UNet2DModel

    dev = pick_device(device)
    unet = UNet2DModel.from_pretrained(os.path.join(model_dir, "unet")).to(dev)
    scheduler = DDPMScheduler.from_pretrained(os.path.join(model_dir, "scheduler"))
    return unet, scheduler


@torch.no_grad()
def generate_images(
    model_dir: str,
    n: int,
    out_dir: str,
    image_size: int = DEFAULT_IMAGE_SIZE,
    upscale_to: int | None = 224,
    batch_size: int = 64,
    num_inference_steps: int = 1000,
    device: str | None = None,
    seed: int = DEFAULT_SEED,
    prefix: str = "synthetic",
) -> list[str]:
    """Sample ``n`` synthetic images from a trained DDPM and save them.

    We run the reverse diffusion loop explicitly (``UNet2DModel`` +
    ``DDPMScheduler``) rather than using ``DDPMPipeline``. This keeps generation
    dependency-light, robust across diffusers/huggingface-hub versions, and
    makes the sampling process legible in the accompanying notebook.

    Args:
        model_dir: directory produced by :func:`train_ddpm`.
        n: number of images to generate.
        out_dir: where to write PNGs.
        upscale_to: if set, bicubically resize each sample to this size (so the
            224px classifier can consume 64px samples). Disclosed limitation.
        num_inference_steps: DDPM denoising steps (1000 = full quality).

    Returns:
        List of written file paths.
    """
    import numpy as np

    dev = pick_device(device)
    unet, scheduler = load_ddpm(model_dir, device=str(dev))
    unet.eval()
    scheduler.set_timesteps(num_inference_steps)

    os.makedirs(out_dir, exist_ok=True)
    gen = torch.Generator(device="cpu").manual_seed(seed)

    paths: list[str] = []
    idx = 0
    remaining = n
    while remaining > 0:
        b = min(batch_size, remaining)
        # Start from Gaussian noise and iteratively denoise.
        sample = torch.randn(
            (b, 3, image_size, image_size), generator=gen
        ).to(dev)
        for t in scheduler.timesteps:
            noise_pred = unet(sample, t).sample
            sample = scheduler.step(noise_pred, t, sample, generator=gen).prev_sample

        # [-1,1] -> [0,255] uint8 NHWC.
        imgs = ((sample.clamp(-1, 1) + 1.0) / 2.0 * 255.0).round().byte()
        imgs = imgs.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        for arr in imgs:
            img = Image.fromarray(arr, mode="RGB")
            if upscale_to and upscale_to != image_size:
                img = img.resize((upscale_to, upscale_to), Image.BICUBIC)
            path = os.path.join(out_dir, f"{prefix}_{idx:05d}.png")
            img.save(path)
            paths.append(path)
            idx += 1
        remaining -= b
        print(f"[diffusion] generated {idx}/{n}")
    return paths


# --------------------------------------------------------------------------- CLI
def _gather_paths(images_dir: str) -> list[str]:
    paths = sorted(
        glob.glob(os.path.join(images_dir, "*.jpg"))
        + glob.glob(os.path.join(images_dir, "*.jpeg"))
        + glob.glob(os.path.join(images_dir, "*.png"))
    )
    if not paths:
        raise SystemExit(f"No images found in {images_dir!r}")
    return paths


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train / sample a DDPM for rare lesion classes.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_tr = sub.add_parser("train", help="train a DDPM on one class folder")
    p_tr.add_argument("--images-dir", required=True, help="folder of that class's JPEGs")
    p_tr.add_argument("--out-dir", required=True)
    p_tr.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    p_tr.add_argument("--epochs", type=int, default=100)
    p_tr.add_argument("--batch-size", type=int, default=32)
    p_tr.add_argument("--lr", type=float, default=1e-4)
    p_tr.add_argument("--base-channels", type=int, default=128)
    p_tr.add_argument("--no-ema", action="store_true")
    p_tr.add_argument("--max-minutes", type=float, default=None)
    p_tr.add_argument("--seed", type=int, default=DEFAULT_SEED)

    p_ge = sub.add_parser("generate", help="sample images from a trained DDPM")
    p_ge.add_argument("--model-dir", required=True)
    p_ge.add_argument("--out-dir", required=True)
    p_ge.add_argument("--n", type=int, required=True)
    p_ge.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    p_ge.add_argument("--upscale-to", type=int, default=224)
    p_ge.add_argument("--batch-size", type=int, default=64)
    p_ge.add_argument("--steps", type=int, default=1000)
    p_ge.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p_ge.add_argument("--prefix", default="synthetic")

    args = parser.parse_args(argv)
    if args.cmd == "train":
        train_ddpm(
            _gather_paths(args.images_dir),
            out_dir=args.out_dir,
            image_size=args.image_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            base_channels=args.base_channels,
            use_ema=not args.no_ema,
            max_minutes=args.max_minutes,
            seed=args.seed,
        )
    elif args.cmd == "generate":
        generate_images(
            model_dir=args.model_dir,
            n=args.n,
            out_dir=args.out_dir,
            image_size=args.image_size,
            upscale_to=args.upscale_to,
            batch_size=args.batch_size,
            num_inference_steps=args.steps,
            seed=args.seed,
            prefix=args.prefix,
        )


if __name__ == "__main__":
    main()
