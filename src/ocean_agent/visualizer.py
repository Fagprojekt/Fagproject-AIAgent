# ===========================
# visualizer.py
# Animated 2-D or 3-D visualisation for OceanWave3D fort.* files
# ===========================
"""
Stand-alone
-----------
python -m ocean_agent.visualizer <data_dir> <output_gif> --fps 5

Inside the agent
----------------
The chat tool calls visualize(data_dir, output_gif, fps)
and auto-detects 2-D vs 3-D from the data.
"""

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path
from typing import List, Tuple

import matplotlib as mpl
mpl.use("Agg")  # head-less backend

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

_FLOAT_RE = re.compile(r"(-?\d+\.\d+)([+-]\d+)")

# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _parse_oceanwave_file(fname: str) -> np.ndarray:
    """Return Nx4 (or Nx3) float array, tolerant of Fortran exp 1.23-45."""
    rows: List[List[float]] = []
    with open(fname, encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split()
            if not parts:
                continue
            parsed: List[float] = []
            for token in parts:
                try:
                    parsed.append(float(token))
                except ValueError:
                    m = _FLOAT_RE.fullmatch(token)
                    parsed.append(float(f"{m.group(1)}E{m.group(2)}") if m else 0.0)
            if len(parsed) >= 3:
                rows.append(parsed[:4])
    return np.asarray(rows, dtype=float)


# --------------------------------------------------------------------------- #
# Main visualiser                                                             #
# --------------------------------------------------------------------------- #
def _detect_grid(first: np.ndarray) -> Tuple[bool, bool, np.ndarray, np.ndarray]:
    """Return is_2d, on_grid, unique_x, unique_y."""
    x, y = first[:, 0], first[:, 1]
    ux, uy = np.unique(x), np.unique(y)
    is_2d = uy.size == 1
    on_grid = not is_2d and (ux.size * uy.size == first.shape[0])
    return is_2d, on_grid, ux, uy


def visualize(data_dir: str | Path,
              output_gif: str | Path,
              fps: int = 5) -> Path:
    data_dir, output_gif = Path(data_dir), Path(output_gif)
    output_gif.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(str(data_dir / "fort.1*")),
                   key=lambda p: int(Path(p).name.split(".")[-1]))
    if not files:
        raise FileNotFoundError(f"No fort.1* files in {data_dir}")

    first = _parse_oceanwave_file(files[0])
    if first.size == 0:
        raise ValueError(f"{files[0]} contains no numeric data")

    is_2d, on_grid, ux, uy = _detect_grid(first)
    x0, y0, z0 = first[:, 0], first[:, 1], first[:, 2]

    # global z-limits (sample ≤20 files)
    samples = [_parse_oceanwave_file(f)[:, 2] for f in files[:20]]
    z_min, z_max = float(np.min(samples)), float(np.max(samples))
    pad = 0.1 * (z_max - z_min or 1)
    z_lim = (z_min - pad, z_max + pad)

    # figure & axis
    plt.rcParams["figure.figsize"] = [12, 8]
    fig = plt.figure()
    if is_2d:
        ax = fig.add_subplot(111)
        sort = np.argsort(x0)
        line, = ax.plot(x0[sort], z0[sort])
        ax.set_xlabel("X"); ax.set_ylabel("η")
        ax.set_ylim(*z_lim)
    else:
        ax = fig.add_subplot(111, projection="3d")
        ax.view_init(elev=30, azim=-60)
        if on_grid:
            X, Y = np.meshgrid(ux, uy)
            surf = ax.plot_surface(X, Y, z0.reshape(uy.size, ux.size),
                                   cmap="viridis", linewidth=0)
        else:
            surf = ax.scatter(x0, y0, z0, c=z0, cmap="viridis")
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="η")
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("η")
        ax.set_zlim(*z_lim)

    # ------------------------------------------------------------------ #
    def _update(i: int):
        data = _parse_oceanwave_file(files[i])
        if data.size == 0:
            return []

        ax.cla()  # clear axis safely

        if is_2d:
            xs, zs = data[:, 0], data[:, 2]
            order = np.argsort(xs)
            ax.plot(xs[order], zs[order])
            ax.set_xlabel("X"); ax.set_ylabel("η")
            ax.set_ylim(*z_lim)
            ax.set_title(f"Frame {i}   {Path(files[i]).name}")
        else:
            xs, ys, zs = data[:, 0], data[:, 1], data[:, 2]
            if on_grid and zs.size == ux.size * uy.size:
                ax.plot_surface(X, Y, zs.reshape(uy.size, ux.size),
                                cmap="viridis", linewidth=0)
            else:
                ax.scatter(xs, ys, zs, c=zs, cmap="viridis")
            ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("η")
            ax.set_zlim(*z_lim)
            ax.view_init(elev=30, azim=-60)
            ax.set_title(f"Frame {i}   {Path(files[i]).name}")
        return []

    ani = FuncAnimation(fig, _update, frames=len(files),
                        interval=1000/fps, blit=False)
    ani.save(output_gif, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return output_gif


# --------------------------------------------------------------------------- #
# CLI entry-point                                                             #
# --------------------------------------------------------------------------- #
def _cli() -> None:
    p = argparse.ArgumentParser(description="Visualise OceanWave3D fort.* files")
    p.add_argument("data_dir"), p.add_argument("output_gif")
    p.add_argument("--fps", type=int, default=5)
    args = p.parse_args()
    out = visualize(args.data_dir, args.output_gif, fps=args.fps)
    print(f"✅ GIF written to {out}")

if __name__ == "__main__":  # pragma: no cover
    _cli()
