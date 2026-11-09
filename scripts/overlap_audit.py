"""
Measure how much overlap actually costs, before trying to fix it.

Overlap is detected without ground truth: a blob of worm pixels carrying far
more medial axis than one body length contains more than one worm. (Junction
count alone is far too loose - skeletonising a single worm leaves head/tail
spurs that register as junctions, which flagged 85% of all blobs on a first
attempt.)

That gives an overlap map per frame, crossed with the pipeline's own output:

  1. How often does overlap happen?
  2. Do detections go missing at overlap sites?
  3. Do TRACKS break at overlap sites?

(3) is the one that matters - 418 tracks for ~24 worms is the blocking problem.
It is measured case/control: sample AT track endpoints, and compare against
continuing tracks in the SAME frames, so worm density is matched between the
two groups.

Usage:
  python scripts/overlap_audit.py <video.mp4> <centerlines.csv> [--n 120]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import ndimage
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects, skeletonize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dtpipe import preprocess as pp

_K = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.uint8)


def overlap_blobs(frame, bg, roi, rel=0.93, min_area=150, body=145.0):
    """Worm blobs, flagged for whether they hold more than one worm."""
    mask = remove_small_objects((frame / np.maximum(bg, 1.0) < rel) & roi,
                                min_area)
    out = []
    for r in regionprops(label(mask)):
        sk = skeletonize(r.image)
        arc = int(sk.sum())
        if arc < 40:
            continue
        nb = ndimage.convolve(sk.astype(np.uint8), _K, mode="constant")
        junctions = int(((nb > 2) & sk).sum())
        out.append(dict(
            cy=r.centroid[0] + r.bbox[0], cx=r.centroid[1] + r.bbox[1],
            arc=arc, junctions=junctions,
            overlapping=bool(arc > 1.55 * body and junctions > 0),
            n_expected=max(1, int(round(arc / body))),
            bbox=r.bbox,
        ))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("csv")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--segments", type=int, default=8)
    ap.add_argument("--near", type=float, default=90.0)
    ap.add_argument("--body", type=float, default=145.0)
    a = ap.parse_args()

    df = pd.read_csv(a.csv, usecols=lambda c: c in
                     ("frame", "time_s", "identity", "x24", "y24"))
    n, fps = pp.video_info(a.video)
    gbg = pp.estimate_background(a.video, n_samples=80)
    bbox = pp.detect_chamber(gbg)
    edges, sbgs = pp.estimate_segment_backgrounds(a.video, a.segments)
    roi = pp.build_roi_mask(gbg, bbox, trim_right=0.10)
    r0, c0, r1, c1 = bbox
    frames = sorted(df.frame.unique())

    cache = {}

    def blobs_at(fi):
        fi = int(fi)
        if fi not in cache:
            k = min(max(int(np.searchsorted(edges, fi, "right") - 1), 0),
                    len(sbgs) - 1)
            raw = pp.read_frames(a.video, [fi])[0]
            cache[fi] = overlap_blobs(raw[r0:r1, c0:c1],
                                      sbgs[k][r0:r1, c0:c1], roi, body=a.body)
        return cache[fi]

    # ---- 1 & 2: overlap prevalence and detection recall ------------------
    pick = [frames[i] for i in
            np.linspace(0, len(frames) - 1, min(a.n, len(frames))).astype(int)]
    tot_blobs = tot_overlap = 0
    det_ov = det_si = exp_ov = exp_si = 0
    for fi in pick:
        sub = df[df.frame == fi]
        pts = (np.stack([sub.x24.to_numpy() - c0, sub.y24.to_numpy() - r0], -1)
               if len(sub) else np.zeros((0, 2)))
        for b in blobs_at(fi):
            tot_blobs += 1
            br0, bc0, br1, bc1 = b["bbox"]
            inside = (((pts[:, 0] >= bc0) & (pts[:, 0] < bc1) &
                       (pts[:, 1] >= br0) & (pts[:, 1] < br1)).sum()
                      if len(pts) else 0)
            if b["overlapping"]:
                tot_overlap += 1
                det_ov += inside
                exp_ov += b["n_expected"]
            else:
                det_si += inside
                exp_si += b["n_expected"]

    print("sampled " + str(len(pick)) + " frames of " + str(len(frames)))
    print("")
    print("1. how often does overlap happen")
    print(f"   worm blobs            {tot_blobs}")
    print(f"   blobs with >1 worm    {tot_overlap} "
          f"({100*tot_overlap/max(tot_blobs,1):.1f}%)")
    print(f"   worms inside them     {exp_ov} of {exp_ov+exp_si} "
          f"({100*exp_ov/max(exp_ov+exp_si,1):.1f}%)")
    print("")

    r_ov = det_ov / max(exp_ov, 1)
    r_si = det_si / max(exp_si, 1)
    print("2. are detections lost at overlap sites")
    print(f"   isolated worms   {det_si}/{exp_si} detected = {100*r_si:.0f}%")
    print(f"   overlapping      {det_ov}/{exp_ov} detected = {100*r_ov:.0f}%")
    print(f"   -> overlap costs {100*(r_si-r_ov):.0f} percentage points")
    print("")

    # ---- 3: case/control at track endpoints ------------------------------
    ends = []
    for tid, g in df.groupby("identity"):
        g = g.sort_values("frame")
        ends.append((int(g.frame.iloc[0]), g.x24.iloc[0], g.y24.iloc[0]))
        ends.append((int(g.frame.iloc[-1]), g.x24.iloc[-1], g.y24.iloc[-1]))
    ends = pd.DataFrame(ends, columns=["frame", "x", "y"])
    lo, hi = min(frames), max(frames)
    cases = ends[(ends.frame > lo + 30) & (ends.frame < hi - 30)]
    cases = cases.sample(min(len(cases), a.n), random_state=0)

    ctrl = []
    for fi in cases.frame.unique():
        here = cases[cases.frame == fi]
        endset = set(zip(here.x.round(1), here.y.round(1)))
        for _, row in df[df.frame == fi].iterrows():
            if (round(row.x24, 1), round(row.y24, 1)) not in endset:
                ctrl.append((fi, row.x24, row.y24))
    ctrl = pd.DataFrame(ctrl, columns=["frame", "x", "y"])
    if len(ctrl) > 4 * len(cases):
        ctrl = ctrl.sample(4 * len(cases), random_state=0)

    def frac_near(tab):
        near = tot = 0
        for _, row in tab.iterrows():
            oc = [(b["cx"] + c0, b["cy"] + r0) for b in blobs_at(row.frame)
                  if b["overlapping"]]
            tot += 1
            if oc:
                d = np.linalg.norm(np.array(oc) - np.array([row.x, row.y]),
                                   axis=1).min()
                near += int(d < a.near)
        return near, tot

    cn, ct = frac_near(cases)
    kn, kt = frac_near(ctrl)
    print("3. do TRACKS break at overlap sites   <- the one that matters")
    print(f"   track endpoints      {cn}/{ct} near an overlap "
          f"= {100*cn/max(ct,1):.0f}%")
    print(f"   continuing tracks    {kn}/{kt} near an overlap "
          f"= {100*kn/max(kt,1):.0f}%   (control, same frames)")
    if kt and kn:
        lift = (cn / max(ct, 1)) / (kn / max(kt, 1))
        print(f"   -> enrichment {lift:.2f}x")
        print("   -> overlap IS driving the breaks" if lift > 1.4
              else "   -> overlap is NOT the main driver of fragmentation")


if __name__ == "__main__":
    main()
