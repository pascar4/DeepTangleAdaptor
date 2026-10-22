"""
Background removal on a single frame, step by step, comparing methods.

Shows, for one frame:
  * the raw crop
  * the global median background vs the per-segment median
  * DIVISION    frame / background
  * SUBTRACTION background - frame
  * each of those thresholded, drawn as a red overlay on the raw frame
  * the final model input

Why it matters which one: in transmitted-light microscopy the signal is
multiplicative (Beer-Lambert - transmitted light is incident light times a
transmission factor), so a worm attenuates by a *fraction* of whatever
illumination reaches it. Division is invariant to that gain; subtraction is
only invariant to an additive offset. The left/right split printed at the end
quantifies the difference across the chamber's illumination gradient.

Usage:
  python scripts/bgdemo.py <video.mp4> --t 6
  python scripts/bgdemo.py <video.mp4> --t 6 --rel-thresh 0.93
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dtpipe import preprocess as pp


def to8(a, lo=None, hi=None):
    a = a.astype(np.float32)
    lo = a.min() if lo is None else lo
    hi = a.max() if hi is None else hi
    return (np.clip((a - lo) / (hi - lo + 1e-9), 0, 1) * 255).astype(np.uint8)


def band(img, text, width, color=False):
    if not color:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h = int(img.shape[0] * width / img.shape[1])
    im = cv2.resize(img, (width, h))
    cv2.rectangle(im, (0, 0), (width, 26), (0, 0, 0), -1)
    cv2.putText(im, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return im


def red_overlay(raw8, mask):
    vis = cv2.cvtColor(raw8, cv2.COLOR_GRAY2BGR)
    vis[mask] = (0, 0, 255)
    return vis


def halves(mask, roi):
    """Flagged fraction in the left vs right half of the ROI."""
    w = mask.shape[1] // 2
    lm, rm = mask[:, :w] & roi[:, :w], mask[:, w:] & roi[:, w:]
    lr, rr = roi[:, :w], roi[:, w:]
    l = 100 * lm.sum() / max(lr.sum(), 1)
    r = 100 * rm.sum() / max(rr.sum(), 1)
    return l, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--t", type=float, default=6.0, help="seconds")
    ap.add_argument("--segments", type=int, default=8)
    ap.add_argument("--rel-thresh", type=float, default=0.93,
                    help="division: flag pixels with frame/bg below this")
    ap.add_argument("--sub-thresh", type=float, default=0.0,
                    help="subtraction: grey levels; 0 = auto-match division")
    ap.add_argument("--out", default="out/bgdemo.png")
    ap.add_argument("--width", type=int, default=1500)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    n, fps = pp.video_info(a.video)
    fi = int(a.t * fps)
    raw_full = pp.read_frames(a.video, [fi])[0]

    global_bg = pp.estimate_background(a.video, n_samples=80)
    bbox = pp.detect_chamber(global_bg)
    r0, c0, r1, c1 = bbox
    edges, seg_bgs = pp.estimate_segment_backgrounds(a.video,
                                                     n_segments=a.segments)
    k = min(max(int(np.searchsorted(edges, fi, "right") - 1), 0),
            len(seg_bgs) - 1)
    unstable = pp.instability_mask(seg_bgs)

    raw = raw_full[r0:r1, c0:c1]
    gbg = global_bg[r0:r1, c0:c1]
    sbg = seg_bgs[k][r0:r1, c0:c1]
    roi_g = pp.build_roi_mask(global_bg, bbox, trim_right=0.10)
    roi_s = pp.build_roi_mask(global_bg, bbox, trim_right=0.10,
                              unstable=unstable)

    print(f"frame {fi}  (t={fi/fps:.2f}s of {n/fps:.0f}s)")
    print(f"chamber rows {r0}..{r1}  cols {c0}..{c1}   "
          f"({c1-c0} x {r1-r0} px)")
    print(f"segment {k+1}/{a.segments} covers "
          f"{edges[k]/fps:.0f}-{edges[k+1]/fps:.0f}s")
    print(f"background level: median {np.median(sbg):.1f} grey, "
          f"left half {np.median(sbg[:, :sbg.shape[1]//2]):.1f}, "
          f"right half {np.median(sbg[:, sbg.shape[1]//2:]):.1f}\n")

    # --- the two methods, on the per-segment background -------------------
    div = raw / np.maximum(sbg, 1.0)          # multiplicative
    sub = sbg - raw                           # additive; worms are darker
    div_g = raw / np.maximum(gbg, 1.0)        # division, global bg, for contrast

    m_div = (div < a.rel_thresh) & roi_s
    # auto-match: pick the subtraction threshold flagging the same pixel count
    if a.sub_thresh > 0:
        st = a.sub_thresh
    else:
        vals = sub[roi_s]
        st = float(np.percentile(vals, 100 * (1 - m_div.sum() / max(roi_s.sum(), 1))))
    m_sub = (sub > st) & roi_s
    m_div_g = (div_g < a.rel_thresh) & roi_g

    print(f"DIVISION   frame / background,  flag < {a.rel_thresh}")
    print(f"  flagged {100*m_div.sum()/roi_s.sum():.2f}% of ROI")
    l, r = halves(m_div, roi_s)
    print(f"  left half {l:.2f}%   right half {r:.2f}%   "
          f"imbalance {abs(l-r):.2f} pp")

    print(f"\nSUBTRACTION   background - frame,  flag > {st:.2f} grey levels")
    print(f"  flagged {100*m_sub.sum()/roi_s.sum():.2f}% of ROI "
          f"(threshold auto-matched to the same count)")
    l2, r2 = halves(m_sub, roi_s)
    print(f"  left half {l2:.2f}%   right half {r2:.2f}%   "
          f"imbalance {abs(l2-r2):.2f} pp")

    print(f"\nagreement: {100*(m_div & m_sub).sum()/max((m_div | m_sub).sum(),1):.1f}% "
          f"IoU between the two masks")
    print(f"  division-only pixels {int((m_div & ~m_sub).sum())}, "
          f"subtraction-only {int((m_sub & ~m_div).sum())}")

    print(f"\nglobal-background division, same threshold: "
          f"{100*m_div_g.sum()/roi_g.sum():.2f}% of ROI flagged "
          f"(vs {100*m_div.sum()/roi_s.sum():.2f}% per-segment)")

    # --- render -----------------------------------------------------------
    W = a.width
    raw8 = to8(raw)
    LO, HI = 0.80, 1.06
    rows = [
        band(raw8, f"1. raw frame, chamber crop  (t={fi/fps:.1f}s)", W),
        band(to8(gbg), "2a. GLOBAL median background (whole recording)", W),
        band(to8(sbg), f"2b. PER-SEGMENT background (segment {k+1}/{a.segments})", W),
        band(to8(div_g, LO, HI), "3a. DIVISION with global bg - bubbles leave residuals", W),
        band(to8(div, LO, HI), "3b. DIVISION with per-segment bg  =  frame / background", W),
        band(to8(sub, -6, 40), "4. SUBTRACTION  =  background - frame", W),
        band(red_overlay(raw8, m_div),
             f"5. DIVISION thresholded  (< {a.rel_thresh})", W, color=True),
        band(red_overlay(raw8, m_sub),
             f"6. SUBTRACTION thresholded  (> {st:.1f} grey, matched count)", W, color=True),
        band(red_overlay(raw8, m_div ^ m_sub),
             "7. DISAGREEMENT between the two thresholded masks", W, color=True),
        band(to8(np.clip(1 - div, 0, None) * roi_s),
             "8. final model input  =  (1 - ratio) x ROI", W),
    ]
    out = np.vstack(rows)
    cv2.imwrite(a.out, out)
    print(f"\n-> {a.out}   ({out.shape[1]}x{out.shape[0]})")


if __name__ == "__main__":
    main()
