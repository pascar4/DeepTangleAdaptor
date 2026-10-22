"""
Decisive experiment: does the released checkpoint detect these worms once the
recording is rescaled into its training domain?

Sweeps the spatial downscale factor, runs detection on one clip per setting,
and writes an overlay of the predicted splines on the original frame.

Usage:
  python scripts/smoke.py <video.mp4> --t 60 --scales 3.0,3.5,4.0
"""
import argparse
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import deeptangle as dt
from dtpipe import preprocess as pp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--model", default="weights/")
    ap.add_argument("--t", type=float, default=60.0, help="seconds")
    ap.add_argument("--scales", default="3.0,3.5,4.0")
    ap.add_argument("--score", type=float, default=0.5)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--trim-right", type=float, default=0.10)
    ap.add_argument("--outdir", default="out")
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)

    n_frames, fps = pp.video_info(a.video)
    stride = pp.recommended_stride(fps)
    print(f"video    {n_frames} frames @ {fps:.2f} fps")
    print(f"stride   {stride}  -> effective {fps/stride:.2f} fps "
          f"(trained at {pp.TRAIN_FPS:.2f})")

    print("estimating background ...")
    bg = pp.estimate_background(a.video, n_samples=80)
    bbox = pp.detect_chamber(bg)
    mask = pp.build_roi_mask(bg, bbox, trim_right=a.trim_right)
    r0, c0, r1, c1 = bbox
    print(f"chamber  rows {r0}..{r1} cols {c0}..{c1}  "
          f"({c1-c0}x{r1-r0}), roi covers {100*mask.mean():.0f}% of it")

    print("loading model ...")
    forward_fn, state = dt.load_model(a.model)

    center = int(a.t * fps)
    idx = pp.clip_indices(center, stride)
    idx = [min(max(i, 0), n_frames - 1) for i in idx]
    raw = pp.read_frames(a.video, idx)

    base = cv2.cvtColor(np.clip(raw[len(raw) // 2], 0, 255).astype(np.uint8),
                        cv2.COLOR_GRAY2BGR)

    for scale in [float(s) for s in a.scales.split(",")]:
        scene = pp.Scene(bg, bbox, mask, fps, n_frames, scale, stride)
        clip = pp.preprocess(raw, scene)

        preds = dt.detect(forward_fn, state, clip[None],
                          threshold=a.score, overlap_threshold=a.overlap)
        n = len(preds.w)
        body = None
        if n:
            present = preds.w[:, 1]                       # (n, 49, 2)
            seg = np.linalg.norm(np.diff(present, axis=1), axis=-1)
            body = float(np.median(seg.sum(axis=1)) * scale)

        print(f"scale {scale:>4.1f}  ->  {n:>3d} detections"
              + (f"   median body {body:.0f} px (full-res)" if body else ""))

        vis = base.copy()
        for k, spline in enumerate(preds.w[:, 1]):
            pts = pp.to_original_coords(spline, scene)[5:-5]
            col = [(0, 200, 0), (0, 165, 255), (255, 120, 0),
                   (0, 0, 230), (200, 0, 200)][k % 5]
            cv2.polylines(vis, [pts.astype(np.int32)], False, col, 2,
                          cv2.LINE_AA)
        cv2.rectangle(vis, (c0, r0), (c1, r1), (120, 120, 120), 1)
        out = os.path.join(a.outdir, f"smoke_s{scale:.1f}_t{int(a.t)}.png")
        cv2.imwrite(out, vis)
        print(f"            -> {out}")


if __name__ == "__main__":
    main()
