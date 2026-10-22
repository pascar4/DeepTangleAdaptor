"""
Run the adapted de(ep)tangle pipeline over a microfluidics recording.

Emits a tidy CSV of centreline coordinates per worm per frame, and optionally
an overlay video.

Usage:
  python scripts/run.py <video.mp4> --duration 60 --out results/
"""
import argparse
import os
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import cv2
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import deeptangle as dt
from dtpipe import postprocess as post
from dtpipe import preprocess as pp

PALETTE = [(66, 133, 244), (219, 68, 55), (15, 157, 88),
           (244, 180, 0), (171, 71, 188), (0, 172, 193)]


def stream_small_frames(video, scene, start, count, chunk=120, verbose=True):
    """Read every `stride`-th frame and return the preprocessed small stack."""
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start))
    out, buf, taken, read_i, chunk_start = [], [], 0, 0, start
    while taken < count:
        ok, fr = cap.read()
        if not ok:
            break
        if read_i % scene.stride == 0:
            if not buf:
                chunk_start = start + read_i
            buf.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32))
            taken += 1
            if len(buf) == chunk:
                out.append(pp.preprocess(
                    np.asarray(buf), scene,
                    bg_override=scene.background_for(chunk_start)))
                buf = []
                if verbose:
                    print(f"\r  preprocessed {taken}/{count}", end="", flush=True)
        read_i += 1
    if buf:
        out.append(pp.preprocess(np.asarray(buf), scene,
                                 bg_override=scene.background_for(chunk_start)))
    cap.release()
    if verbose:
        print(f"\r  preprocessed {taken}/{count}   ")
    return np.concatenate(out) if out else np.zeros((0,))


def predict_all(forward_fn, state, clips, batch=8, verbose=True):
    """Batched forward pass; returns a list of per-frame Predictions."""
    preds = []
    n = len(clips)
    t0 = time.time()
    for i in range(0, n, batch):
        chunk = jnp.asarray(clips[i:i + batch])
        p = dt.predict(forward_fn, state, chunk)
        p = jax.tree_util.tree_map(np.asarray, p)
        for k in range(len(chunk)):
            preds.append(dt.Predictions(p.w[k], p.s[k], p.p[k]))
        if verbose and (i // batch) % 10 == 0:
            done = min(i + batch, n)
            rate = done / max(time.time() - t0, 1e-6)
            print(f"\r  inferred {done}/{n}  ({rate:.1f} clip/s, "
                  f"eta {(n-done)/max(rate,1e-6):.0f}s)", end="", flush=True)
    if verbose:
        print(f"\r  inferred {n}/{n}  in {time.time()-t0:.0f}s        ")
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--model", default="weights/")
    ap.add_argument("--out", default="results")
    ap.add_argument("--start", type=float, default=0.0, help="seconds")
    ap.add_argument("--duration", type=float, default=0.0,
                    help="seconds; 0 = whole recording")
    ap.add_argument("--scale", type=float, default=0.0,
                    help="spatial downscale; 0 = auto from measured body length")
    ap.add_argument("--body-px", type=float, default=145.0,
                    help="measured full-res body length, used when --scale=0")
    ap.add_argument("--score", type=float, default=0.5)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--nms-radius", type=float, default=8.0,
                    help="geometric NMS radius, model-space px")
    ap.add_argument("--trim-right", type=float, default=0.10)
    ap.add_argument("--trim-left", type=float, default=0.0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--video-out", action="store_true")
    ap.add_argument("--segments", type=int, default=8,
                    help="background segments; 1 = single global median")
    ap.add_argument("--instability-mask", action="store_true",
                    help="also exclude pixels whose background varies across "
                         "segments. Off by default: per-segment backgrounds "
                         "already remove the bubble false positives, and the "
                         "mask additionally deletes real worms near bubbles "
                         "(verified against hand annotation on frame 205).")
    ap.add_argument("--min-align", type=float, default=0.90,
                    help="geometric NMS: tangent alignment required to call "
                         "two overlapping splines the same worm")
    ap.add_argument("--no-refine", action="store_true",
                    help="skip medial-axis tip refinement")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    stem = os.path.splitext(os.path.basename(a.video))[0]

    n_frames, fps = pp.video_info(a.video)
    stride = pp.recommended_stride(fps)
    scale = a.scale or pp.recommended_scale(a.body_px)

    print(f"video    {stem}")
    print(f"         {n_frames} frames @ {fps:.2f} fps")
    print(f"scale    {scale:.2f}x  (body {a.body_px:.0f} -> "
          f"{a.body_px/scale:.0f} px)")
    print(f"stride   {stride}  -> {fps/stride:.2f} fps "
          f"(trained {pp.TRAIN_FPS:.2f})")

    print("background ...")
    bg = pp.estimate_background(a.video, n_samples=80)
    bbox = pp.detect_chamber(bg)
    seg_edges = seg_bgs = unstable = None
    if a.segments > 1:
        seg_edges, seg_bgs = pp.estimate_segment_backgrounds(
            a.video, n_segments=a.segments)
        print(f"         {a.segments} segment backgrounds")
        if a.instability_mask:
            unstable = pp.instability_mask(seg_bgs)
            print(f"         instability mask excludes "
                  f"{100*unstable.mean():.1f}% of the frame")
    mask = pp.build_roi_mask(bg, bbox, trim_right=a.trim_right,
                             trim_left=a.trim_left, unstable=unstable)
    scene = pp.Scene(bg, bbox, mask, fps, n_frames, scale, stride,
                     seg_edges, seg_bgs)
    r0, c0, r1, c1 = bbox
    print(f"         chamber {c1-c0}x{r1-r0} px, roi {100*mask.mean():.0f}%")

    start_f = int(a.start * fps)
    avail = (n_frames - start_f) // stride
    want = int(a.duration * fps / stride) if a.duration else avail
    n_small = min(want, avail)

    print(f"reading  {n_small} decimated frames from t={a.start:.0f}s ...")
    small = stream_small_frames(a.video, scene, start_f, n_small)
    if len(small) < pp.NFRAMES + 1:
        sys.exit("not enough frames")
    print(f"         model space {small.shape[2]}x{small.shape[1]}")

    clips = dt.video_to_clips(small, num_frames=pp.NFRAMES)
    print(f"         {len(clips)} clips")

    print("loading model ...")
    forward_fn, state = dt.load_model(a.model)

    print("inference ...")
    raw_preds = predict_all(forward_fn, state, clips, batch=a.batch)

    print("suppression ...")
    cleaned, n_lat, n_geo = [], 0, 0
    for p in raw_preds:
        keep = dt.non_max_suppression(p, threshold=a.score,
                                      overlap_threshold=a.overlap, cutoff=48)
        p = jax.tree_util.tree_map(lambda x: x[keep], p)
        n_lat += len(p.w)
        g = post.geometric_nms(p, tol=a.nms_radius / 2.0,
                               min_overlap=0.45,
                               min_align=a.min_align)
        p = jax.tree_util.tree_map(lambda x: x[g], p)
        n_geo += len(p.w)
        cleaned.append(p)
    nf = max(len(cleaned), 1)
    print(f"         latent NMS {n_lat/nf:.1f}/frame -> "
          f"+geometric {n_geo/nf:.1f}/frame")

    print("tracking ...")
    ids, splines = dt.identity_assignment(cleaned)

    # Flip canonicalisation, per identity, using the predicted future slice.
    by_id = {}
    for t, (fr_ids, fr_sp) in enumerate(zip(ids, splines)):
        for j, tid in enumerate(fr_ids):
            by_id.setdefault(tid, []).append((t, j))
    for tid, entries in by_id.items():
        pres = [splines[t][j] for t, j in entries]
        fut = [np.asarray(cleaned[t].w[j, 2]) for t, j in entries]
        fixed, _ = post.canonicalize_flips(pres, fut)
        for (t, j), sp in zip(entries, fixed):
            splines[t][j] = sp

    model_bounds = (0, 0, small.shape[1], small.shape[2])
    ids, splines, stats = post.stitch_tracks(ids, splines, model_bounds)
    n_ids = len({t for fr in ids for t in fr})
    per_frame = np.mean([len(fr) for fr in ids]) if ids else 0
    print(f"         {n_ids} tracks, {per_frame:.1f} worms/frame, "
          f"merged {stats['merged']}, "
          f"same-frame dropped {stats['same_frame_dropped']}")

    # --- tip refinement -----------------------------------------------
    # Splines are converted to full-resolution coordinates first: the medial
    # axis is far better resolved at native scale than in model space.
    orig_by_frame = []
    for t, fr_sp in enumerate(splines):
        orig_by_frame.append(pp.to_original_coords(np.asarray(fr_sp), scene)
                             if len(fr_sp) else np.zeros((0, 49, 2)))

    if not a.no_refine:
        print("refining tips against medial axis ...")
        cap = cv2.VideoCapture(str(a.video))
        n_ok = n_try = 0
        pos = -1
        for t in range(len(orig_by_frame)):
            if len(orig_by_frame[t]) == 0:
                continue
            fi = start_f + (t + pp.NFRAMES // 2) * stride
            if fi != pos + 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            pos = fi
            if not ok:
                break
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            m = post.segmentation_mask(g, scene.background_for(fi))
            out = []
            for sp in orig_by_frame[t]:
                r, good = post.refine_to_medial_axis(sp, m)
                n_try += 1
                n_ok += int(good)
                out.append(r)
            orig_by_frame[t] = np.asarray(out)
        cap.release()
        print(f"         refined {n_ok}/{n_try} splines "
              f"({100*n_ok/max(n_try,1):.0f}%); rest kept the model spline")

    print("writing ...")
    rows = []
    npts = pp.NFRAMES  # placeholder, replaced below
    for t, (fr_ids, fr_sp) in enumerate(zip(ids, splines)):
        if len(fr_sp) == 0:
            continue
        for tid, sp in zip(fr_ids, orig_by_frame[t]):
            frame_idx = start_f + (t + pp.NFRAMES // 2) * stride
            row = {"frame": frame_idx,
                   "time_s": frame_idx / fps,
                   "identity": int(tid)}
            for k, (x, y) in enumerate(sp):
                row[f"x{k}"] = round(float(x), 2)
                row[f"y{k}"] = round(float(y), 2)
            rows.append(row)
    df = pd.DataFrame(rows)
    csv = os.path.join(a.out, f"{stem}_centerlines.csv")
    df.to_csv(csv, index=False)
    print(f"         {csv}  ({len(df)} rows)")

    if a.video_out and len(ids):
        path = os.path.join(a.out, f"{stem}_overlay.mp4")
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps / stride, (1920, 480))
        cap = cv2.VideoCapture(str(a.video))
        for t, (fr_ids, fr_sp) in enumerate(zip(ids, splines)):
            fi = start_f + (t + pp.NFRAMES // 2) * stride
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok:
                break
            if len(fr_sp):
                for tid, sp in zip(fr_ids, orig_by_frame[t]):
                    col = PALETTE[int(tid) % len(PALETTE)]
                    cv2.polylines(frame, [sp[5:-5].astype(np.int32)], False,
                                  col, 2, cv2.LINE_AA)
                    cv2.circle(frame, tuple(sp[5].astype(int)), 4, col, -1)
                    cv2.putText(frame, str(int(tid)),
                                tuple(sp[5].astype(int) + np.array([6, -6])),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
            vw.write(frame)
        cap.release()
        vw.release()
        print(f"         {path}")

    print("done.")


if __name__ == "__main__":
    main()
