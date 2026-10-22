"""
Preprocessing that maps a microfluidics recording into the domain the released
de(ep)tangle checkpoint was trained on.

The checkpoint expects:
  * worms BRIGHT on a dark ground (the reference scripts do `255 - x`)
  * bodies 30-45 px long
  * 11 frames spanning ~0.55 s, i.e. ~18 fps
  * roughly unit-scaled intensities

A 1920x480 @30fps chamber recording with ~145 px worms satisfies none of these,
so every step below exists to close one specific gap.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects

# 11 snapshots over 0.55 s -> linspace spacing of 0.055 s.
TRAIN_FPS = 10.0 / 0.55
TRAIN_BODY_PX = (30.0, 45.0)
NFRAMES = 11


@dataclass
class Scene:
    """Everything about a recording that preprocessing needs, measured once."""
    background: np.ndarray          # (H, W) float32, full frame
    bbox: tuple[int, int, int, int]  # chamber: (r0, c0, r1, c1)
    roi_mask: np.ndarray            # (h, w) bool, True = analyse, cropped to bbox
    fps: float
    n_frames: int
    scale: float                    # spatial downscale factor
    stride: int                     # temporal decimation
    seg_edges: np.ndarray = None    # (n_seg+1,) frame indices
    seg_bgs: np.ndarray = None      # (n_seg, H, W) per-segment backgrounds

    def background_for(self, frame_idx):
        """Background valid at `frame_idx` - per-segment when available."""
        if self.seg_bgs is None:
            return self.background
        k = int(np.searchsorted(self.seg_edges, frame_idx, "right") - 1)
        return self.seg_bgs[min(max(k, 0), len(self.seg_bgs) - 1)]

    @property
    def crop_shape(self):
        r0, c0, r1, c1 = self.bbox
        return (r1 - r0, c1 - c0)


def read_frames(path, indices):
    """Read specific frame indices as float32 grayscale."""
    cap = cv2.VideoCapture(str(path))
    out = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if not ok:
            cap.release()
            raise IOError(f"could not read frame {i} of {path}")
        out.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32))
    cap.release()
    return np.asarray(out)


def video_info(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"could not open {path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return n, fps


def estimate_background(path, n_samples=80):
    """Median over frames spread across the whole recording.

    Deliberately global rather than rolling: during a seizure the worms stop
    moving, and a rolling median would absorb a paralysed worm into the
    background - erasing exactly the phenotype being measured.  The cost is
    that slowly growing bubbles are only partly captured; mask them instead.
    """
    n, _ = video_info(path)
    idx = np.linspace(0, n - 1, min(n_samples, n)).astype(int)
    return np.median(read_frames(path, idx), axis=0)


def estimate_segment_backgrounds(path, n_segments=8, per_segment=25):
    """Per-segment medians, so a slowly growing bubble is tracked.

    A single global median holds whatever configuration dominates the
    recording, which leaves huge residuals at bubble edges in the frames that
    differ from it - and the detector fires on those residuals.

    Still a median *within* each segment rather than a rolling one: during a
    seizure the worms stop moving, and a short rolling window would absorb a
    paralysed worm into the background, erasing the phenotype being measured.
    Segments are long enough that a worm occupies any given pixel for a small
    fraction of the frames, yet short enough that bubbles are quasi-static.
    """
    n, _ = video_info(path)
    edges = np.linspace(0, n, n_segments + 1).astype(int)
    bgs = []
    for a, b in zip(edges[:-1], edges[1:]):
        idx = np.linspace(a, max(a, b - 1), per_segment).astype(int)
        bgs.append(np.median(read_frames(path, idx), axis=0))
    return edges, np.asarray(bgs)


def instability_mask(seg_bgs, tol=6.0):
    """Pixels whose background changes across segments - i.e. not static scene.

    Growing bubbles light this up; walls, debris and illumination do not.
    Excluding these costs a little chamber area and removes the dominant source
    of false positives.
    """
    return seg_bgs.std(axis=0) > tol


def detect_chamber(background, min_area=5000):
    """Bounding box of the bright chamber interior."""
    norm = (background - background.min()) / (np.ptp(background) + 1e-9)
    bright = remove_small_objects(norm > 0.55, min_area)
    lab = label(bright)
    if lab.max() == 0:
        return (0, 0, *background.shape)
    return max(regionprops(lab), key=lambda r: r.area).bbox


def build_roi_mask(background, bbox, trim_right=0.0, trim_left=0.0,
                   erode_px=6, unstable=None):
    """Analysis mask inside the chamber, with optional end trims.

    `trim_right`/`trim_left` are fractions of chamber width to exclude - use
    them to drop the bubble foam that accumulates at an electrode end.
    """
    r0, c0, r1, c1 = bbox
    h, w = r1 - r0, c1 - c0
    norm = (background - background.min()) / (np.ptp(background) + 1e-9)
    inside = (norm > 0.55)[r0:r1, c0:c1]
    if erode_px > 0:
        k = np.ones((erode_px, erode_px), np.uint8)
        inside = cv2.erode(inside.astype(np.uint8), k).astype(bool)
    if trim_left > 0:
        inside[:, : int(w * trim_left)] = False
    if trim_right > 0:
        inside[:, int(w * (1 - trim_right)) :] = False
    if unstable is not None:
        u = unstable[r0:r1, c0:c1]
        u = cv2.dilate(u.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        inside &= ~u
    return inside


def recommended_scale(body_px, target=None):
    """Downscale factor that lands `body_px` in the trained body-length range."""
    lo, hi = TRAIN_BODY_PX
    # 36 px measured optimal on this rig (length ratio 0.99 vs 0.89 at 37.5)
    target = target if target is not None else 36.0
    return float(body_px) / target


def recommended_stride(fps):
    """Temporal decimation that brings `fps` closest to the trained rate."""
    cands = range(1, 6)
    return min(cands, key=lambda k: abs(np.log((fps / k) / TRAIN_FPS)))


def clip_indices(center, stride, n_frames=NFRAMES):
    """Frame indices of an n_frames window centred on `center`."""
    half = n_frames // 2
    return [center + stride * (i - half) for i in range(n_frames)]


def preprocess(frames, scene, pad_to=16, bg_override=None):
    """Raw frames -> model-ready clip.

    Returns (clip, meta) where clip is (n_frames, H', W') float32 with worms
    bright, and meta carries what is needed to map coordinates back.
    """
    r0, c0, r1, c1 = scene.bbox
    base = scene.background if bg_override is None else bg_override
    bg = base[r0:r1, c0:c1]
    mask = scene.roi_mask

    # 1. Background DIVISION, not subtraction: illumination drift and the
    #    lamp/electrode transients are multiplicative, and the static chamber
    #    walls divide out exactly.
    x = frames[:, r0:r1, c0:c1] / np.maximum(bg, 1.0)

    # 2. Worms are darker than their surround -> make them bright and positive.
    x = np.clip(1.0 - x, 0.0, None)

    # 3. Kill everything outside the analysable chamber region.
    x *= mask[None]

    # 4. Per-FRAME normalisation. The reference scripts run CLAHE over the whole
    #    stack, which couples frames together and whose temporal kernel depends
    #    on how many frames were loaded; per-frame scaling is flicker-immune.
    hi = np.percentile(x.reshape(len(x), -1), 99.5, axis=1)
    x = x / (hi[:, None, None] + 1e-6)
    x = np.clip(x, 0.0, 2.0)

    # 5. Spatial downscale so bodies land in the trained 30-45 px range.
    if scene.scale != 1.0:
        h, w = x.shape[1:]
        nh, nw = max(1, round(h / scene.scale)), max(1, round(w / scene.scale))
        x = np.stack([cv2.resize(f, (nw, nh), interpolation=cv2.INTER_AREA)
                      for f in x])

    # 6. Pad to a multiple of the backbone stride so the prediction grid is
    #    exact and reproducible across videos.
    if pad_to > 1:
        h, w = x.shape[1:]
        ph, pw = (-h) % pad_to, (-w) % pad_to
        if ph or pw:
            x = np.pad(x, ((0, 0), (0, ph), (0, pw)))

    return x.astype(np.float32)


def to_original_coords(pts, scene):
    """Map (..., 2) xy from model space back to full-frame pixels."""
    r0, c0, _, _ = scene.bbox
    out = np.asarray(pts, dtype=np.float64) * scene.scale
    out[..., 0] += c0
    out[..., 1] += r0
    return out
