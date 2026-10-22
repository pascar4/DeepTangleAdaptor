"""
Quantify centreline accuracy against a simple intensity segmentation, and
search for a systematic coordinate offset.

Usage: python scripts/validate.py <video.mp4> <centerlines.csv>
"""
import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dtpipe import preprocess as pp


def worm_mask(frame, bg, rel=0.93):
    return (frame / (bg + 1e-6)) < rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("csv")
    ap.add_argument("--n", type=int, default=25, help="frames to check")
    a = ap.parse_args()

    df = pd.read_csv(a.csv)
    xs = [c for c in df.columns if c.startswith("x")]
    ys = [c for c in df.columns if c.startswith("y")]

    bg = pp.estimate_background(a.video, n_samples=80)
    frames = sorted(df["frame"].unique())
    pick = [frames[i] for i in np.linspace(0, len(frames) - 1, a.n).astype(int)]

    base, shifted = [], {}
    offsets = [(dx, dy) for dx in range(-14, 15, 2) for dy in range(-14, 15, 2)]

    for fi in pick:
        raw = pp.read_frames(a.video, [fi])[0]
        m = worm_mask(raw, bg)
        # distance from every pixel to the nearest worm pixel
        dist = ndimage.distance_transform_edt(~m)
        sub = df[df["frame"] == fi]
        pts = np.stack([sub[xs].to_numpy(), sub[ys].to_numpy()], -1
                       ).reshape(-1, 2)
        pts = pts[5:-5] if len(pts) > 10 else pts

        for dx, dy in offsets:
            q = pts + np.array([dx, dy])
            c = np.clip(q[:, 0], 0, dist.shape[1] - 1).astype(int)
            r = np.clip(q[:, 1], 0, dist.shape[0] - 1).astype(int)
            shifted.setdefault((dx, dy), []).append(float(np.median(dist[r, c])))
        c = np.clip(pts[:, 0], 0, dist.shape[1] - 1).astype(int)
        r = np.clip(pts[:, 1], 0, dist.shape[0] - 1).astype(int)
        base.append(float(np.median(dist[r, c])))

    means = {k: float(np.mean(v)) for k, v in shifted.items()}
    best = min(means, key=means.get)

    print(f"frames checked       {len(pick)}")
    print(f"median dist to worm  {np.mean(base):.2f} px   (no correction)")
    print(f"best global offset   dx={best[0]:+d}  dy={best[1]:+d}"
          f"   -> {means[best]:.2f} px")
    print()
    print("residual vs offset (px), dy rows / dx cols")
    dxs = sorted({k[0] for k in means})
    dys = sorted({k[1] for k in means})
    print("      " + "".join(f"{dx:>6}" for dx in dxs))
    for dy in dys:
        print(f"{dy:>5} " + "".join(f"{means[(dx,dy)]:>6.1f}" for dx in dxs))


if __name__ == "__main__":
    main()
