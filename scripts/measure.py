"""
Measure a recording's geometry so the deeptangle preprocessing can be configured:
chamber bounds, worm body length (px), worms per frame, and the resulting
downscale factor needed to land worms in the model's 30-45 px training range.

Usage:  python scripts/measure.py <video.mp4> [--n 40]
"""
import argparse
import sys

import cv2
import numpy as np
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects, skeletonize

# The released checkpoint was trained on bodies of this length, in pixels.
TRAIN_BODY_PX = (30.0, 45.0)

def is_worm(r, arc, min_arc=60, max_arc=260, min_w=3.0, max_w=18.0, max_sol=0.85):
    """A worm is a long thin low-solidity blob. Width must be area/arc:
    a curved worm's ellipse minor axis measures the bend, not the body."""
    if arc <= 0:
        return False
    width = r.area / arc
    return (min_arc <= arc <= max_arc
            and min_w <= width <= max_w
            and r.solidity < max_sol)



def sample_frames(path, n):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit(f"could not open {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    idx = np.linspace(0, total - 1, n).astype(int)
    frames = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if ok:
            frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
    cap.release()
    return np.asarray(frames, dtype=np.float32), fps, total, idx


def chamber_bounds(bg):
    """Locate the bright interior of the chamber against the darker surround."""
    norm = (bg - bg.min()) / (np.ptp(bg) + 1e-9)
    bright = norm > 0.55
    bright = remove_small_objects(bright, 5000)
    lab = label(bright)
    if lab.max() == 0:
        return (0, 0, bg.shape[0], bg.shape[1])
    biggest = max(regionprops(lab), key=lambda r: r.area)
    return biggest.bbox  # (min_row, min_col, max_row, max_col)


def worm_stats(frame, bg, bbox, rel_thresh=0.90, min_area=120):
    """Background-divide, threshold dark objects, and measure each blob."""
    r0, c0, r1, c1 = bbox
    f = frame[r0:r1, c0:c1]
    b = bg[r0:r1, c0:c1]
    ratio = f / (b + 1e-6)
    mask = ratio < rel_thresh
    mask = remove_small_objects(mask, min_area)
    out = []
    for r in regionprops(label(mask)):
        sk = skeletonize(r.image)
        arc = int(sk.sum())
        ecc = r.eccentricity
        out.append(
            dict(
                is_worm=is_worm(r, arc),
                area=r.area,
                arc=arc,
                major=r.axis_major_length,
                minor=r.axis_minor_length,
                ecc=ecc,
                solidity=r.solidity,
            )
        )
    return out, mask.shape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--n", type=int, default=40, help="frames to sample")
    ap.add_argument("--rel-thresh", type=float, default=0.90)
    ap.add_argument("--min-area", type=int, default=120)
    a = ap.parse_args()

    frames, fps, total, idx = sample_frames(a.video, a.n)
    print(f"video      {a.video}")
    print(f"frames     {total} @ {fps:.3f} fps  ({total/fps:.1f} s)")
    print(f"resolution {frames.shape[2]} x {frames.shape[1]}")
    print(f"sampled    {len(frames)} frames\n")

    bg = np.median(frames, axis=0)
    bbox = chamber_bounds(bg)
    r0, c0, r1, c1 = bbox
    print(f"chamber    rows {r0}..{r1}  cols {c0}..{c1}   "
          f"({c1-c0} x {r1-r0} px)\n")

    per_frame_counts = []
    arcs, majors, minors, areas = [], [], [], []
    for fr in frames:
        blobs, _ = worm_stats(fr, bg, bbox, a.rel_thresh, a.min_area)
        # Worm-shaped: elongated and not a filled disc (bubbles are round/solid).
        worms = [b for b in blobs if b["is_worm"]]
        per_frame_counts.append(len(worms))
        for b in worms:
            arcs.append(b["arc"])
            majors.append(b["major"])
            minors.append(b["minor"])
            areas.append(b["area"])

    c = np.array(per_frame_counts)
    print("worm-like objects per frame")
    print(f"  mean {c.mean():.1f}   median {np.median(c):.0f}   "
          f"min {c.min()}   max {c.max()}\n")

    if not arcs:
        sys.exit("no worm-like objects found - adjust --rel-thresh / --min-area")

    arcs = np.array(arcs, dtype=float)
    q = np.percentile(arcs, [10, 25, 50, 75, 90])
    print("skeleton arc length (px)  <- body length")
    print(f"  p10 {q[0]:.0f}  p25 {q[1]:.0f}  median {q[2]:.0f}  "
          f"p75 {q[3]:.0f}  p90 {q[4]:.0f}")
    print(f"  mean width (minor axis) {np.mean(minors):.1f} px\n")

    body = q[2]
    lo = body / TRAIN_BODY_PX[1]
    hi = body / TRAIN_BODY_PX[0]
    rec = (lo + hi) / 2
    print(f"downscale needed to reach the {TRAIN_BODY_PX[0]:.0f}-"
          f"{TRAIN_BODY_PX[1]:.0f} px training range")
    print(f"  factor range {lo:.2f}x .. {hi:.2f}x   recommended {rec:.2f}x")
    print(f"  -> frame {frames.shape[2]}x{frames.shape[1]} becomes "
          f"{round(frames.shape[2]/rec)}x{round(frames.shape[1]/rec)}")
    print(f"  -> body length becomes {body/rec:.0f} px")

    # Temporal: the checkpoint was trained at 11 snapshots over 0.55 s.
    train_fps = 10 / 0.55
    print(f"\ntemporal sampling")
    print(f"  checkpoint trained at {train_fps:.2f} fps "
          f"(11 frames over 0.55 s)")
    print(f"  video is {fps:.2f} fps -> "
          f"decimate by {fps/train_fps:.2f}x to match")
    for k in (1, 2, 3):
        eff = fps / k
        print(f"    keep every {k}: {eff:.2f} fps "
              f"({eff/train_fps:.2f}x the trained rate)")


if __name__ == "__main__":
    main()
