"""
Compare predicted centrelines against the worm's true medial axis, and test
whether the checkpoint's eigenworm basis can even represent these postures.

Three questions, in order of how much they cost to fix:
  1. Does the spline follow the body's centreline?      (bidirectional distance)
  2. Does it span the whole body?                       (length ratio, coverage)
  3. CAN a 12-component eigenworm basis express this posture at all?
     (projection residual - if this fails, nothing downstream can succeed)

Usage: python scripts/shapecheck.py <video.mp4> <centerlines.csv> --t 60
"""
import argparse
import os
import sys
from collections import deque

import cv2
import numpy as np
import pandas as pd
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects, skeletonize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dtpipe import preprocess as pp

NEIGH = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def order_skeleton(sk):
    """Order skeleton pixels into a path from one endpoint to the other."""
    pts = {(int(r), int(c)) for r, c in zip(*np.nonzero(sk))}
    if len(pts) < 5:
        return None
    adj = {p: [q for q in ((p[0] + dr, p[1] + dc) for dr, dc in NEIGH)
               if q in pts] for p in pts}
    ends = [p for p, n in adj.items() if len(n) == 1]

    def farthest(src):
        seen, order = {src: None}, deque([src])
        last = src
        while order:
            cur = order.popleft()
            last = cur
            for q in adj[cur]:
                if q not in seen:
                    seen[q] = cur
                    order.append(q)
        return last, seen

    start = ends[0] if ends else next(iter(pts))
    a, _ = farthest(start)
    b, parents = farthest(a)
    path = []
    cur = b
    while cur is not None:
        path.append(cur)
        cur = parents[cur]
    return np.array([(c, r) for r, c in path], dtype=float)  # xy


def resample(curve, n=49):
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))]
    if d[-1] <= 0:
        return None
    t = np.linspace(0, d[-1], n)
    return np.stack([np.interp(t, d, curve[:, i]) for i in (0, 1)], axis=-1)


def arclen(c):
    return float(np.linalg.norm(np.diff(c, axis=0), axis=1).sum())


def mean_min_dist(a, b):
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    return float(d.min(axis=1).mean())


def pca_residual(curve49, A, scale):
    """Project a measured centreline through the checkpoint's eigenworm basis."""
    c = np.asarray(curve49, float) / scale        # into model pixel scale
    c = c - c.mean(axis=0)
    best = None
    for cand in (c, c[::-1]):
        x = cand.reshape(-1)
        recon = (x @ A.T) @ A
        rms = float(np.sqrt(np.mean((x - recon) ** 2)))
        best = rms if best is None else min(best, rms)
    return best                                    # model px per coordinate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("csv")
    ap.add_argument("--t", type=float, default=60.0)
    ap.add_argument("--scale", type=float, default=3.5)
    ap.add_argument("--rel", type=float, default=0.93)
    ap.add_argument("--outdir", default="out/shape")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    A = np.load("weights/eigenworms_transform.npy")
    df = pd.read_csv(a.csv)
    xs = [f"x{i}" for i in range(49)]
    ys = [f"y{i}" for i in range(49)]

    _, fps = pp.video_info(a.video)
    target = int(a.t * fps)
    frame_idx = int(df["frame"].iloc[(df["frame"] - target).abs().argmin()])
    sub = df[df["frame"] == frame_idx]
    print(f"frame {frame_idx} (t={frame_idx/fps:.1f}s), "
          f"{len(sub)} predicted worms\n")

    bg = pp.estimate_background(a.video, n_samples=80)
    raw = pp.read_frames(a.video, [frame_idx])[0]
    mask = remove_small_objects((raw / (bg + 1e-6)) < a.rel, 150)

    vis = cv2.cvtColor(np.clip(raw, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    preds = [np.stack([r[xs].to_numpy(float), r[ys].to_numpy(float)], -1)
             for _, r in sub.iterrows()]
    ids = sub["identity"].tolist()

    rows = []
    for r in regionprops(label(mask)):
        arc0 = int(skeletonize(r.image).sum())
        width = r.area / max(arc0, 1)
        if not (60 <= arc0 <= 260 and 3.0 <= width <= 18.0 and r.solidity < .85):
            continue
        sk = order_skeleton(skeletonize(r.image))
        if sk is None:
            continue
        r0, c0 = r.bbox[0], r.bbox[1]
        sk = sk + np.array([c0, r0])
        truth = resample(sk, 49)
        if truth is None:
            continue

        # nearest prediction by midpoint
        if not preds:
            continue
        mids = np.array([p[24] for p in preds])
        k = int(np.argmin(np.linalg.norm(mids - truth[24], axis=1)))
        pred = preds[k]
        if np.linalg.norm(mids[k] - truth[24]) > 60:
            continue

        # orientation-agnostic comparison
        d_fwd = mean_min_dist(pred, truth)
        d_rev = mean_min_dist(pred[::-1], truth)
        p2t = min(d_fwd, d_rev)
        t2p = mean_min_dist(truth, pred)
        Lp, Lt = arclen(pred), arclen(truth)
        cov = float((np.linalg.norm(
            truth[:, None, :] - pred[None, :, :], axis=-1).min(axis=1) < 6).mean())
        res = pca_residual(truth, A, a.scale)

        rows.append(dict(id=ids[k], pred2truth=p2t, truth2pred=t2p,
                         Lpred=Lp, Ltruth=Lt, ratio=Lp / Lt,
                         coverage=cov, pca_rms_model=res,
                         pca_rms_full=res * a.scale))

        cv2.polylines(vis, [truth.astype(np.int32)], False, (0, 220, 0), 1,
                      cv2.LINE_AA)
        cv2.polylines(vis, [pred.astype(np.int32)], False, (0, 80, 255), 1,
                      cv2.LINE_AA)

        # zoomed crop
        pad = 30
        x0 = int(max(0, min(truth[:, 0].min(), pred[:, 0].min()) - pad))
        x1 = int(min(vis.shape[1], max(truth[:, 0].max(), pred[:, 0].max()) + pad))
        y0 = int(max(0, min(truth[:, 1].min(), pred[:, 1].min()) - pad))
        y1 = int(min(vis.shape[0], max(truth[:, 1].max(), pred[:, 1].max()) + pad))
        crop = vis[y0:y1, x0:x1]
        if crop.size:
            crop = cv2.resize(crop, None, fx=4, fy=4,
                              interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(os.path.join(a.outdir, f"worm_{ids[k]}.png"), crop)

    if not rows:
        sys.exit("no matched worms")

    t = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("green = measured medial axis, orange = predicted spline\n")
    print(t.round(2).to_string(index=False))
    print()
    print(f"pred->truth  mean {t.pred2truth.mean():.2f} px   "
          f"(how far the spline sits off the true centreline)")
    print(f"truth->pred  mean {t.truth2pred.mean():.2f} px   "
          f"(how much of the body the spline misses)")
    print(f"length ratio mean {t.ratio.mean():.2f}    "
          f"coverage@6px mean {t.coverage.mean():.2f}")
    print()
    print(f"eigenworm basis residual  {t.pca_rms_model.mean():.2f} model px "
          f"= {t.pca_rms_full.mean():.1f} full-res px")
    print("  ^ floor imposed by the 12-component basis: no detector tuning "
          "can beat this")
    cv2.imwrite(os.path.join(a.outdir, "all.png"), vis)
    print(f"\n-> {a.outdir}/")


if __name__ == "__main__":
    main()
