"""Frame rendering: PNG fields, GIF/MP4 assembly, particle plots.

matplotlib is used for annotated figures; PIL for fast raw-image frames and GIF assembly.
MP4 export shells out to ``ffmpeg`` when it is on ``PATH`` and degrades gracefully to GIF
when it is not (see ``docs/DESIGN.md``).

All images are drawn with **y pointing up**: arrays are flipped vertically before being
handed to the raster image so that row 0 (smallest y) appears at the bottom of the image.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from mackerel.grid import Array, BoolArray  # noqa: F401  (BoolArray re-exported for callers)

__all__ = [
    "OUTPUTS",
    "ffmpeg_available",
    "render_particles",
    "save_field_png",
    "save_frames_gif",
    "save_frames_mp4",
    "save_image",
]

RGBArray = np.ndarray[tuple[int, int, int], np.dtype[np.uint8]]

OUTPUTS = Path("outputs")


def ffmpeg_available() -> bool:
    """True when an ``ffmpeg`` binary is on ``PATH``."""
    return shutil.which("ffmpeg") is not None


def _to_uint8_rgb(field: Array, vmin: float | None, vmax: float | None, cmap: str) -> RGBArray:
    """Normalise a 2-D field to uint8 RGB, y-up."""
    import matplotlib

    cm = matplotlib.colormaps[cmap]
    f = np.asarray(field, dtype=np.float64)
    lo = float(np.nanmin(f)) if vmin is None else vmin
    hi = float(np.nanmax(f)) if vmax is None else vmax
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        norm = np.zeros_like(f)
    else:
        norm = (f - lo) / (hi - lo)
    rgb = cm(np.clip(norm, 0.0, 1.0))[:, :, :3]
    # y-up: flip rows so the smallest y ends up at the bottom of the image
    return np.ascontiguousarray((np.flipud(rgb) * 255.0).astype(np.uint8))


def save_image(rgb: RGBArray, path: str | Path) -> Path:
    """Write a uint8 RGB array (already y-up flipped) to PNG."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(p)
    return p


def save_field_png(
    field: Array,
    path: str | Path,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    scale: int = 4,
) -> Path:
    """Save a 2-D scalar field as a colour-mapped PNG, y pointing up."""
    rgb = _to_uint8_rgb(field, vmin, vmax, cmap)
    if scale > 1:
        rgb = np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)
    return save_image(rgb, path)


def save_frames_gif(frames: list[RGBArray], path: str | Path, duration_ms: int = 60) -> Path:
    """Assemble y-up RGB frames (uint8 ``(H, W, 3)``) into an animated GIF with Pillow."""
    if not frames:
        raise ValueError("no frames to write")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.fromarray(np.ascontiguousarray(f)).convert("P", palette=Image.Palette.ADAPTIVE)
              for f in frames]
    images[0].save(p, save_all=True, append_images=images[1:], duration=duration_ms, loop=0)
    return p


def save_frames_mp4(frames: list[RGBArray], path: str | Path, fps: int = 30) -> Path | None:
    """Encode y-up RGB frames to MP4 with ``ffmpeg``; returns ``None`` if ffmpeg is absent."""
    if not ffmpeg_available():
        return None
    if not frames:
        raise ValueError("no frames to write")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    h, w, _ = frames[0].shape
    ffmpeg = shutil.which("ffmpeg")
    assert ffmpeg is not None
    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-pix_fmt", "yuv420p", str(p),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    for f in frames:
        proc.stdin.write(np.ascontiguousarray(f).astype(np.uint8).tobytes())
    proc.stdin.close()
    proc.wait()
    return p


def render_particles(
    px: Array,
    py: Array,
    grid_shape: tuple[int, int],
    path: str | Path,
    color: tuple[int, int, int] = (70, 160, 255),
    background: tuple[int, int, int] = (8, 10, 16),
    scale: int = 4,
) -> Path:
    """Draw a particle scatter (y-up) onto a raster image and save it."""
    ny, nx = grid_shape
    img: RGBArray = np.zeros((ny, nx, 3), dtype=np.uint8)
    img[:, :] = background
    i = np.clip((np.asarray(px) * nx).astype(np.intp), 0, nx - 1)
    j = np.clip((np.asarray(py) * ny).astype(np.intp), 0, ny - 1)
    img[j, i] = color
    img = np.flipud(img)
    if scale > 1:
        img = np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)
    return save_image(img, path)
