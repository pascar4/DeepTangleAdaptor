"""
Dump an annotated frame showing exactly what the segmentation calls a worm,
so the filters can be tuned by eye instead of by guesswork.

Usage: python scripts/overlay.py <video.mp4> --t 60 --out out.png
"""
import argparse

import cv2
import numpy as np
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects, skeletonize

def is_worm(r, arc, min_arc=60, max_arc=260, min_w=3.0, max_w=18.0, max_sol=0.85):
    """A worm is a long thin low-solidity blob. Width must be area/arc:
    a curved worm's ellipse minor axis measures the bend, not the body."""
    if arc <= 0:
        return False
    width = r.area / arc
    return (min_arc <= arc <= max_arc
            and min_w <= width <= max_w
            and r.solidity < max_sol)



def build_background(path, n=60):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in np.linspace(0, total - 1, n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if ok:
            frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
    cap.release()
    return np.median(np.asarray(frames, np.float32), axis=0), total


def grab(path, t, fps=30.0):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
    ok, fr = cap.read()
    cap.release()
    return cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--t", type=float, default=60)
    ap.add_argument("--out", default="overlay.png")
    ap.add_argument("--rel-thresh", type=float, default=0.90)
    ap.add_argument("--min-area", type=int, default=120)
    ap.add_argument("--max-minor", type=float, default=22.0)
    ap.add_argument("--min-arc", type=int, default=40)
    ap.add_argument("--max-arc", type=int, default=400)
    a = ap.parse_args()

    bg, _ = build_background(a.video)
    fr = grab(a.video, a.t)

    ratio = fr / (bg + 1e-6)
    mask = remove_small_objects(ratio < a.rel_thresh, a.min_area)

    vis = cv2.cvtColor(np.clip(fr, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    kept = rejected = 0
    for r in regionprops(label(mask)):
        arc = int(skeletonize(r.image).sum())
        worm = is_worm(r, arc, a.min_arc, a.max_arc)
        r0, c0, r1, c1 = r.bbox
        if worm:
            kept += 1
            cv2.rectangle(vis, (c0, r0), (c1, r1), (0, 200, 0), 2)
            cv2.putText(vis, f"{arc}/{r.area/max(arc,1):.1f}", (c0, max(12, r0 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 1)
        else:
            rejected += 1
            cv2.rectangle(vis, (c0, r0), (c1, r1), (0, 0, 220), 1)
            cv2.putText(vis, f"{arc}/{r.area/max(arc,1):.1f}/{r.solidity:.2f}",
                        (c0, max(12, r0 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 220), 1)

    cv2.imwrite(a.out, vis)
    print(f"t={a.t}s   kept(green)={kept}   rejected(red)={rejected}   -> {a.out}")


if __name__ == "__main__":
    main()
