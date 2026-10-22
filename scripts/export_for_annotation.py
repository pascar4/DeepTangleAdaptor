"""
Export frames for hand annotation, and read the annotations back.

Export writes background-removed, contrast-stretched PNGs - the view where the
worms read most clearly - at 1:1 pixels of the chamber crop, so any mark drawn
on them maps straight back to original frame coordinates.

To annotate: open each PNG in any editor (Paint, Preview, GIMP) and mark every
worm in a SATURATED COLOUR - red, green, blue, magenta, anything that is not
grey. The underlying image is greyscale, so any coloured pixel is a mark.

  * a dot or blob per worm  -> gives a count and a position
  * a traced line along the body -> also gives a centreline to score shape against

Save as PNG, same filename, into the same folder. Then:

  python scripts/export_for_annotation.py <video> --read annot/

Usage:
  python scripts/export_for_annotation.py <video.mp4> --n 12 --out annot/
  python scripts/export_for_annotation.py <video.mp4> --read annot/
"""
import argparse
import glob
import os
import re
import sys

import cv2
import numpy as np
import pandas as pd
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dtpipe import preprocess as pp


def build_scene(video, segments=8):
    n, fps = pp.video_info(video)
    gbg = pp.estimate_background(video, n_samples=80)
    bbox = pp.detect_chamber(gbg)
    edges, sbgs = pp.estimate_segment_backgrounds(video, n_segments=segments)
    unstable = pp.instability_mask(sbgs)
    roi = pp.build_roi_mask(gbg, bbox, trim_right=0.10, unstable=unstable)
    return pp.Scene(gbg, bbox, roi, fps, n, 4.0, 2, edges, sbgs), unstable


def enhanced(video, scene, frame_idx, show_roi=True, unstable=None):
    r0, c0, r1, c1 = scene.bbox
    raw = pp.read_frames(video, [frame_idx])[0]
    bg = scene.background_for(frame_idx)
    ratio = raw[r0:r1, c0:c1] / np.maximum(bg[r0:r1, c0:c1], 1.0)
    sig = np.clip(1.0 - ratio, 0, None)
    # stretch on a fixed window so brightness is comparable across frames
    img = np.clip(sig / 0.45, 0, 1)
    img = (255 * (1.0 - img)).astype(np.uint8)      # dark worms on white: easier to draw on
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if show_roi and unstable is not None:
        u = unstable[r0:r1, c0:c1]
        vis[u] = (235, 235, 235)                     # excluded region, flat grey
    return vis


def do_export(a):
    scene, unstable = build_scene(a.video, a.segments)
    os.makedirs(a.out, exist_ok=True)
    r0, c0, r1, c1 = scene.bbox
    idx = np.linspace(scene.n_frames * 0.02, scene.n_frames * 0.97,
                      a.n).astype(int)
    for fi in idx:
        vis = enhanced(a.video, scene, int(fi), unstable=unstable)
        if a.scale != 1.0:
            vis = cv2.resize(vis, None, fx=a.scale, fy=a.scale,
                             interpolation=cv2.INTER_CUBIC)
        name = f"frame_{int(fi):06d}_t{fi/scene.fps:07.2f}s.png"
        cv2.imwrite(os.path.join(a.out, name), vis)
    meta = pd.DataFrame([{"bbox_r0": r0, "bbox_c0": c0,
                          "bbox_r1": r1, "bbox_c1": c1,
                          "fps": scene.fps, "n_frames": scene.n_frames,
                          "video": os.path.basename(a.video)}])
    meta.to_csv(os.path.join(a.out, "_meta.csv"), index=False)
    print(f"exported {len(idx)} frames to {a.out}/")
    print(f"  chamber crop is {c1-c0} x {r1-r0} px, offset (+{c0}, +{r0})")
    print(f"  worms are DARK on white; flat grey areas are excluded from analysis")
    print("\nmark every worm in a saturated colour, save as PNG in place, then:")
    print(f"  python scripts/export_for_annotation.py \"{a.video}\" --read {a.out}")


def extract_marks(path, min_sat=60, min_px=8):
    """Coloured pixels on a greyscale image are the annotation."""
    img = cv2.imread(path)
    if img is None:
        return None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.int16)
    val = hsv[:, :, 2].astype(np.int16)
    marked = (sat > min_sat) & (val > 40)
    out = []
    for r in regionprops(label(marked)):
        if r.area < min_px:
            continue
        sk = skeletonize(r.image)
        arc = int(sk.sum())
        ys, xs = np.nonzero(r.image)
        out.append(dict(
            area=int(r.area), arc=arc,
            cx=float(r.centroid[1] + r.bbox[1]),
            cy=float(r.centroid[0] + r.bbox[0]),
            elongated=bool(arc > 25 and r.area / max(arc, 1) < 14),
        ))
    return out


def do_read(a):
    meta = pd.read_csv(os.path.join(a.read, "_meta.csv")).iloc[0]
    c0, r0 = int(meta.bbox_c0), int(meta.bbox_r0)
    rows = []
    files = sorted(glob.glob(os.path.join(a.read, "frame_*.png")))
    if not files:
        sys.exit(f"no frame_*.png in {a.read}")
    print(f"{'frame':>8} {'t (s)':>8} {'marks':>6} {'traced':>7}")
    for f in files:
        m = re.search(r"frame_(\d+)_t", os.path.basename(f))
        if not m:
            continue
        fi = int(m.group(1))
        marks = extract_marks(f)
        if marks is None:
            continue
        n_line = sum(1 for k in marks if k["elongated"])
        print(f"{fi:>8} {fi/meta.fps:>8.2f} {len(marks):>6} {n_line:>7}")
        for k in marks:
            rows.append(dict(frame=fi, time_s=fi / meta.fps,
                             x=k["cx"] + c0, y=k["cy"] + r0,
                             arc=k["arc"], traced=k["elongated"]))
    if not rows:
        print("\nno coloured marks found - was the image saved as PNG "
              "(not re-saved as greyscale)?")
        return
    df = pd.DataFrame(rows)
    out = os.path.join(a.read, "ground_truth.csv")
    df.to_csv(out, index=False)
    print(f"\n{len(df)} marks over {df.frame.nunique()} frames -> {out}")
    print(f"mean {df.groupby('frame').size().mean():.1f} worms/frame")


def do_read_image(a):
    """Read one annotated image, whatever size it came back at."""
    scene, unstable = build_scene(a.video, a.segments)
    r0, c0, r1, c1 = scene.bbox
    crop_w, crop_h = c1 - c0, r1 - r0

    img = cv2.imread(a.read_image)
    if img is None:
        sys.exit(f"could not read {a.read_image}")
    h, w = img.shape[:2]
    # the annotated copy may have been resized anywhere along the way
    sx, sy = crop_w / w, crop_h / h
    print(f"annotated image {w}x{h}  ->  chamber crop {crop_w}x{crop_h}"
          f"   (scale {sx:.3f}, {sy:.3f})")
    if abs(sx / max(sy, 1e-9) - 1) > 0.05:
        print("  ! aspect ratio differs from the crop - marks may be skewed")

    marks = extract_marks_img(img)
    if not marks:
        sys.exit("no saturated-colour marks found. Mark the worms in a strong "
                 "colour (pure red/green/blue/magenta) and save as PNG.")

    rows = []
    for k in marks:
        rows.append(dict(
            frame=a.frame if a.frame is not None else -1,
            x=k["cx"] * sx + c0, y=k["cy"] * sy + r0,
            arc_px=k["arc"] * (sx + sy) / 2,
            traced=k["elongated"]))
    df = pd.DataFrame(rows)
    n_traced = int(df.traced.sum())
    print("")
    print(f"found {len(df)} marks ({n_traced} traced as lines, "
          f"{len(df)-n_traced} as blobs/dots)")
    print(df.round(1).to_string(index=False))
    out = a.read_image.rsplit(".", 1)[0] + "_marks.csv"
    df.to_csv(out, index=False)
    print("")
    print(f"-> {out}")


def extract_marks_img(img, min_sat=100, min_val=50, min_px=150,
                      min_hue_sep=8, join_gap=1e9, verbose=True):
    """Separate annotations by colour, then by connectivity.

    Connectivity alone is not enough: where two worms cross, the marks drawn on
    them touch and merge into one component.  Annotators use a different colour
    per worm precisely so they stay distinguishable.

    Hue peaks rather than k-means: k-means has to be told how many clusters to
    find - which is the very thing being measured - and it splits single strokes
    at their antialiased edges.  Peaks in the hue histogram of the saturated
    stroke cores are self-determining and stable.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].astype(np.int16)
    sat = hsv[:, :, 1].astype(np.int16)
    val = hsv[:, :, 2].astype(np.int16)
    core = (sat > min_sat) & (val > min_val)      # stroke cores only
    if core.sum() < min_px:
        return []

    hist = np.bincount(hue[core], minlength=180).astype(float)
    hist = np.convolve(np.r_[hist[-4:], hist, hist[:4]],
                       np.ones(5) / 5, mode="same")[4:-4]   # circular smooth

    peaks = []
    for h in np.argsort(hist)[::-1]:
        if hist[h] < 0.04 * hist.max():
            break
        if all(min(abs(h - q), 180 - abs(h - q)) >= min_hue_sep for q in peaks):
            peaks.append(int(h))
    if verbose:
        print(f"  hue peaks: {sorted(peaks)}  ({len(peaks)} colours)")

    # assign every marked pixel (looser mask) to its nearest peak hue
    marked = (sat > 60) & (val > 40)
    ys, xs = np.nonzero(marked)
    hv = hue[ys, xs]
    d = np.stack([np.minimum(np.abs(hv - q), 180 - np.abs(hv - q))
                  for q in peaks], axis=1)
    owner = d.argmin(axis=1)

    out = []
    for i, q in enumerate(peaks):
        m = np.zeros(marked.shape, np.uint8)
        sel = owner == i
        m[ys[sel], xs[sel]] = 1
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        # One colour = one worm.  Where another mark crosses this one it
        # occludes a few pixels and splits the stroke, so re-join fragments of
        # the same hue that sit close together before measuring.
        lab = label(m.astype(bool))
        comps = [r for r in regionprops(lab) if r.area >= min_px // 4]
        if not comps:
            continue
        groups = []
        for r in sorted(comps, key=lambda r: -r.area):
            placed = False
            for g in groups:
                if any(_bbox_gap(r.bbox, o.bbox) < join_gap for o in g):
                    g.append(r)
                    placed = True
                    break
            if not placed:
                groups.append([r])
        for g in groups:
            area = sum(r.area for r in g)
            if area < min_px:
                continue
            gm = np.zeros(m.shape, bool)
            for r in g:
                gm[lab == r.label] = True
            arc = int(skeletonize(gm).sum())
            yy, xx = np.nonzero(gm)
            out.append(dict(area=int(area), arc=arc, hue=q,
                            n_frag=len(g),
                            cx=float(xx.mean()), cy=float(yy.mean()),
                            elongated=bool(arc > 25 and area / max(arc, 1) < 30)))
    return out


def _bbox_gap(a, b):
    """Gap in px between two (min_row, min_col, max_row, max_col) boxes."""
    dr = max(a[0] - b[2], b[0] - a[2], 0)
    dc = max(a[1] - b[3], b[1] - a[3], 0)
    return float(np.hypot(dr, dc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--out", default="annot")
    ap.add_argument("--read", default=None,
                    help="read annotated PNGs back from this folder")
    ap.add_argument("--segments", type=int, default=8)
    ap.add_argument("--read-image", default=None,
                    help="a single annotated image (e.g. pasted into chat)")
    ap.add_argument("--frame", type=int, default=None,
                    help="frame index that --read-image corresponds to")
    ap.add_argument("--scale", type=float, default=2.0,
                    help="export magnification (marks are mapped back by it)")
    a = ap.parse_args()
    if a.read_image:
        do_read_image(a)
    elif a.read:
        do_read(a)
    else:
        do_export(a)


if __name__ == "__main__":
    main()
