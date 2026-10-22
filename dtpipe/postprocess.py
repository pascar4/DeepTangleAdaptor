"""
Detection cleanup and track assembly, adapted for sparse well-separated worms.

de(ep)tangle's latent-space NMS exists because entangled worms on a dense plate
share a centre of mass, so geometry cannot separate them.  In a microfluidics
chamber with a handful of well-separated animals that situation barely arises,
while the latent codes - trained on 30-45 px synthetic bodies - generalise
poorly enough to leave two surviving detections on one worm.  Geometric NMS on
the predicted splines is the better tool for this regime.
"""
from __future__ import annotations

from collections import defaultdict

import cv2
import numpy as np


def _unit_tangents(c):
    t = np.gradient(np.asarray(c, float), axis=0)
    n = np.linalg.norm(t, axis=1, keepdims=True)
    return t / np.maximum(n, 1e-9)


def _overlap_and_alignment(a, b, tol):
    """How much two splines coincide, and whether they run along each other.

    Returns (overlap_fraction, mean_|cos| between tangents where they coincide).

    Overlap alone cannot tell a duplicate from a genuine crossing: two
    detections of ONE worm lie along the same curve, while two worms crossing
    also share points - just transversally, near the intersection only.  The
    tangent alignment separates them.  Hand-annotated frame 205 has a crossing
    pair 37 px apart that overlap-only suppression collapsed into one track.
    """
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    ja, da = d.argmin(axis=1), d.min(axis=1)
    jb, db = d.argmin(axis=0), d.min(axis=0)
    ca, cb = da < tol, db < tol
    frac = max(ca.mean(), cb.mean())
    if not ca.any() and not cb.any():
        return frac, 0.0
    ta, tb = _unit_tangents(a), _unit_tangents(b)
    cos = []
    if ca.any():
        cos.append(np.abs(np.sum(ta[ca] * tb[ja[ca]], axis=1)))
    if cb.any():
        cos.append(np.abs(np.sum(tb[cb] * ta[jb[cb]], axis=1)))
    return frac, float(np.concatenate(cos).mean())


def geometric_nms(preds, tol=5.0, min_overlap=0.45, min_align=0.80,
                  radius=None):
    """Suppress a lower-scoring detection lying ALONG a better one.

    A detection is a duplicate only when it both coincides with a survivor
    (`min_overlap` of its length within `tol` model px) and runs parallel to it
    there (`min_align` mean |cos| between tangents).  A worm crossing another
    satisfies the first test but not the second, so it survives - which is the
    case de(ep)tangle's latent space exists to handle.
    """
    if radius is not None:                      # old signature
        tol = radius / 2.0
    w, s = np.asarray(preds.w), np.asarray(preds.s)
    order = np.argsort(s)[::-1]
    keep, chosen = [], []
    for i in order:
        cur = w[i, 1]
        dup = False
        for j in chosen:
            frac, cos = _overlap_and_alignment(cur, w[j, 1], tol)
            if frac >= min_overlap and cos >= min_align:
                dup = True
                break
        if not dup:
            keep.append(i)
            chosen.append(i)
    mask = np.zeros(len(w), bool)
    mask[keep] = True
    return mask


def canonicalize_flips(splines_by_frame, futures_by_frame):
    """Make point ordering consistent along a track.

    de(ep)tangle's linking metric evaluates min(d, d_flipped) and discards
    which one won, so the emitted splines have arbitrary per-frame ordering.
    Anchor on the model's own predicted FUTURE slice rather than the previous
    present slice - it is a better reference when a worm reorients quickly.
    """
    out = [splines_by_frame[0]]
    flipped = [False]
    for i in range(1, len(splines_by_frame)):
        ref = futures_by_frame[i - 1]          # predicted future at i-1
        if flipped[-1]:
            ref = ref[::-1]
        cur = splines_by_frame[i]
        f = np.sum((ref - cur[::-1]) ** 2) < np.sum((ref - cur) ** 2)
        out.append(cur[::-1] if f else cur)
        flipped.append(f)
    return out, flipped


def _endpoints(tracks, values, bounds, window=5, pad=12):
    """Track starts/ends that happen away from the chamber edge."""
    r0, c0, r1, c1 = bounds

    def near_edge(spline):
        x, y = spline[:, 0], spline[:, 1]
        return bool((x < c0 + pad).any() or (x > c1 - pad).any()
                    or (y < r0 + pad).any() or (y > r1 - pad).any())

    res = {"start": [], "end": []}
    for i in range(window, len(tracks) - window):
        past, future = set(), set()
        for j in range(1, window + 1):
            past.update(tracks[i - j])
            future.update(tracks[i + j])
        for j, tid in enumerate(tracks[i]):
            sp = values[i][j]
            if near_edge(sp):
                continue
            mid = sp[len(sp) // 2]
            if tid not in past:
                res["start"].append((i, tid, mid[0], mid[1]))
            elif tid not in future:
                res["end"].append((i, tid, mid[0], mid[1]))
    return res


def stitch_tracks(tracks, values, bounds, max_gap=45, max_jump=60,
                  min_len=8):
    """Rejoin identities broken by a worm drifting out of the focal plane.

    Replaces de(ep)tangle's `merge_tracks`, which assumes a square frame and a
    single scalar border padding.  Defaults are far more permissive than the
    originals (8 frames / 20 px) because z-dropouts in a chamber last much
    longer than an occlusion on a flat plate.
    """
    ends = _endpoints(tracks, values, bounds)
    end, start = np.array(ends["end"]), np.array(ends["start"])

    # Frames in which each identity is present, so two tracks that coexist are
    # never merged.  Without this guard a track ending near where an unrelated
    # worm's track begins gets absorbed, and every frame where both were
    # present then loses one detection.
    present = defaultdict(set)
    for i, fr in enumerate(tracks):
        for t in fr:
            present[t].add(i)

    pairs = {}
    if len(end) and len(start):
        dt_ = start[None, :, 0] - end[:, None, 0]
        dx = np.linalg.norm(start[None, :, 2:] - end[:, None, 2:], axis=-1)
        ok = (dt_ > 0) & (dt_ <= max_gap) & (dx <= max_jump)
        cand = sorted(zip(*np.nonzero(ok)),
                      key=lambda ij: dx[ij[0], ij[1]])
        for i, j in cand:
            a, b = int(end[i, 1]), int(start[j, 1])
            a, b = pairs.get(a, a), pairs.get(b, b)
            if a == b:
                continue
            if present[a] & present[b]:          # they coexist - not the same worm
                continue
            lo, hi = sorted((a, b))
            pairs[hi] = lo
            present[lo] |= present[hi]

    def resolve(t):
        seen = set()
        while t in pairs and t not in seen:
            seen.add(t)
            t = pairs[t]
        return t

    relabelled = [[resolve(t) for t in fr] for fr in tracks]

    counts = defaultdict(int)
    for fr in relabelled:
        for t in fr:
            counts[t] += 1

    out_t, out_v, dropped = [], [], 0
    for i, fr in enumerate(relabelled):
        seen, tt, vv = set(), [], []
        for j, t in enumerate(fr):
            if counts[t] < min_len:
                continue
            if t in seen:           # same id twice in one frame after merging
                dropped += 1
                continue
            seen.add(t)
            tt.append(t)
            vv.append(values[i][j])
        out_t.append(tt)
        out_v.append(np.asarray(vv))
    return out_t, out_v, {"merged": len(pairs), "same_frame_dropped": dropped}


# ---------------------------------------------------------------------------
# Tip refinement
#
# Measured against a segmented medial axis, the model's splines track the
# mid-body to ~2.6 px but the outermost 5 points at each end to ~9.2 px - which
# is why deeptangle's own scripts plot x[5:-5].  The eigenworm basis is not the
# limit (it reconstructs a measured centreline to ~0.4 px); the detector simply
# regresses tips toward smooth swimming postures and cuts tight terminal hooks.
#
# So: keep deeptangle for what it is good at - finding worms and holding their
# identity through contact - and snap the geometry onto the segmentation's
# medial axis whenever that axis is unambiguous.  When worms touch, the blob
# merges and its axis is meaningless; there we keep the model's spline, which
# is exactly the case deeptangle was built for.
# ---------------------------------------------------------------------------

from skimage.measure import label as _label
from skimage.morphology import skeletonize as _skel

_NEIGH = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _order_skeleton(sk):
    pts = {(int(r), int(c)) for r, c in zip(*np.nonzero(sk))}
    if len(pts) < 5:
        return None
    adj = {p: [q for q in ((p[0] + dr, p[1] + dc) for dr, dc in _NEIGH)
               if q in pts] for p in pts}

    def farthest(src):
        seen, queue, last = {src: None}, [src], src
        while queue:
            cur = queue.pop(0)
            last = cur
            for q in adj[cur]:
                if q not in seen:
                    seen[q] = cur
                    queue.append(q)
        return last, seen

    start = next((p for p, n in adj.items() if len(n) == 1), next(iter(pts)))
    a, _ = farthest(start)
    b, parents = farthest(a)
    path, cur = [], b
    while cur is not None:
        path.append(cur)
        cur = parents[cur]
    return np.array([(c, r) for r, c in path], float)


def _resample(curve, n=49):
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))]
    if d[-1] <= 0:
        return None
    t = np.linspace(0, d[-1], n)
    return np.stack([np.interp(t, d, curve[:, i]) for i in (0, 1)], -1)


def _refine_once(spline, mask, corridor, max_len_ratio, max_shift):
    h, w = mask.shape
    poly = np.round(spline).astype(np.int32)
    if poly[:, 0].max() < 0 or poly[:, 1].max() < 0:
        return spline, False

    corr = np.zeros((h, w), np.uint8)
    cv2.polylines(corr, [poly], False, 1, thickness=2 * int(corridor))
    local = mask & corr.astype(bool)
    if local.sum() < 30:
        return spline, False

    lab = _label(local)
    mid = spline[len(spline) // 2]
    ci, ri = int(round(mid[0])), int(round(mid[1]))
    comp_id = 0
    if 0 <= ri < h and 0 <= ci < w:
        comp_id = lab[ri, ci]
    if comp_id == 0:
        rr = slice(max(0, ri - 6), min(h, ri + 7))
        cc = slice(max(0, ci - 6), min(w, ci + 7))
        vals = lab[rr, cc][lab[rr, cc] > 0]
        if vals.size == 0:
            return spline, False
        comp_id = int(np.bincount(vals).argmax())

    axis = _order_skeleton(_skel(lab == comp_id))
    if axis is None:
        return spline, False
    refined = _resample(axis, len(spline))
    if refined is None:
        return spline, False

    model_len = float(np.linalg.norm(np.diff(spline, axis=0), axis=1).sum())
    axis_len = float(np.linalg.norm(np.diff(refined, axis=0), axis=1).sum())
    if model_len <= 0 or not (0.55 <= axis_len / model_len <= max_len_ratio):
        return spline, False

    if np.sum((refined - spline) ** 2) > np.sum((refined[::-1] - spline) ** 2):
        refined = refined[::-1]

    if float(np.linalg.norm(refined - spline, axis=1).mean()) > max_shift:
        return spline, False
    return refined, True


def refine_to_medial_axis(spline, mask, corridors=(20, 36, 54),
                          max_len_ratio=1.8, max_shift=26.0):
    """Snap one spline onto the medial axis of the worm it sits on.

    The blob is isolated inside a corridor around the spline first: worms touch
    the chamber wall and debris constantly, and the skeleton of a merged blob
    wanders (measured length ratios of 1.6-3.6 and mean shifts of 17-35 px).
    The corridor is a capsule, so it also extends past each spline end, which
    is what lets a missed tip be recovered.

    The corridor is widened progressively: when the model spline badly misses a
    terminal hook, a tight corridor never sees the hook at all.  A wider one
    risks swallowing a neighbour, so the length-ratio and shift gates still
    apply at every width, and the first width that passes wins.

    Returns (refined, ok); on any doubt the input spline is returned unchanged.
    """
    # Take the BEST accepted corridor, not the first.  A tight corridor that
    # clipped a tail still passes the gates (its axis is simply shorter), and
    # first-accept then blocks the wider corridor that would have recovered the
    # missing body.  Measured: worm 2 at frame 1860 accepted at c=20 with 142 px
    # where c=36 gives the full 158 px.
    best, best_len = None, -1.0
    for c in corridors:
        out, ok = _refine_once(spline, mask, c, max_len_ratio, max_shift)
        if not ok:
            continue
        ln = float(np.linalg.norm(np.diff(out, axis=0), axis=1).sum())
        if ln > best_len:
            best, best_len = out, ln
    return (best, True) if best is not None else (spline, False)


def segmentation_mask(frame, background, rel_thresh=0.93, min_area=60):
    """Dark-object mask used for refinement, in the same frame as the splines."""
    from skimage.morphology import remove_small_objects
    return remove_small_objects((frame / (background + 1e-6)) < rel_thresh,
                                min_area)
