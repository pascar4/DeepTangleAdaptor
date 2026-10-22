# de(ep)tangle for microfluidics seizure assays

Adapts the released [de(ep)tangle](https://github.com/kirkegaardlab/deeptangle)
checkpoint to the `C_elegans_video_dataset` chamber recordings and extracts
per-worm centrelines to CSV.

Status: **detection, tracking and centreline geometry working.** Absolute
head/tail polarity is not implemented yet (see Limitations).

## Quick start

```bash
.venv/Scripts/python.exe scripts/run.py "path/to/video.mp4" --out results --video-out
```

Whole folder:

```bash
.venv/Scripts/python.exe scripts/batch.py "G:/My Drive/C_elegans_video_dataset/Cropped/Main/Videos" --out results --skip-existing
```

About 8 min per 343 s recording on CPU, so the 23-video set is roughly 3 hours.

## Why adaptation is needed

The checkpoint was trained on synthetic plates whose statistics differ from this
rig in four measurable ways. Each preprocessing step closes one gap.

| Gap | Checkpoint expects | These recordings | Fix |
|---|---|---|---|
| Body length | 30-45 px | **~145 px** | downscale **4.0x** (measured optimum) |
| Frame rate | ~18.2 fps (11 frames / 0.55 s) | 30 fps | keep every 2nd frame -> 15 fps |
| Polarity | worms bright on dark | worms dark on light | background division, then `1 - ratio` |
| Clutter | random isotropic blobs | thin wall lines, growing bubble foam | fixed-background division + eroded chamber ROI, right end trimmed |

Measure any new rig with `scripts/measure.py`; it reports chamber bounds,
body-length percentiles, worms per frame, and the recommended scale and stride.

## Pipeline

1. **Per-segment backgrounds** (`--segments`, default 8) plus an
   **instability mask**. See "Background handling" below - a single global
   median is not sufficient here.
2. **Chamber detection + ROI mask** - largest bright region, eroded, with an
   optional right-end trim (`--trim-right`, default 0.10) for the electrode
   bubble foam.
3. **Background division**, not subtraction. Illumination drift is
   multiplicative and the static walls divide out exactly.
4. **Per-frame normalisation** - replaces the reference scripts' stack-wide
   CLAHE, whose temporal kernel depends on how many frames were loaded and so
   normalises `detect.py` and `track.py` differently.
5. **Downscale + pad** to a multiple of the backbone stride (16).
6. **11-frame clips**, stride 1, batched inference through `lax.scan`.
7. **Latent NMS** (deeptangle's own) **then geometric NMS**.
8. **Tip refinement** against the segmentation's medial axis.
9. **Flip canonicalisation** per track, anchored on the model's predicted
   *future* slice.
10. **Track stitching** with chamber-aware bounds and a co-occurrence guard,
    replacing `merge_tracks` (which assumes a square frame and one scalar pad).

## Background handling

A single global median **does not work on these recordings**. The bubbles grow
over the session, so the median holds whatever configuration dominates, and
frames that differ from it get large residuals at bubble edges. The detector
fires on those residuals: in the first 20 s, **18.8% of all detections were on
bubble rims**, with tracks following the rim of the big right bubble and the
left bubble edge.

Two changes fix it:

- **Per-segment medians.** The recording is split into `--segments` chunks and a
  median is taken within each, so a slowly growing bubble is tracked. It is
  still a median *within* a segment rather than a rolling one: during a seizure
  the worms stop moving, and a short rolling window would absorb a paralysed
  worm into the background, erasing the phenotype being measured. Segments are
  long enough that a worm occupies any pixel for a small fraction of frames, yet
  short enough that bubbles are quasi-static.
- **Instability mask.** Pixels whose background varies across segments are
  non-static scene - growing bubbles light this up, walls and debris do not.
  Those regions are dilated and excluded.

Result on the first 20 s: detections in the right-25% bubble zone went
**18.8% -> 0.0%**.

**The cost is real.** The instability mask excludes ~9.8% of the frame, so the
ROI drops from 79% to 64% and worms that enter those regions are lost. On the
50-90 s window this took detections from 6.2 to 5.7 per frame and raised track
count from 9 to 11 (more fragmentation, since a worm crossing the excluded zone
breaks its track). Use `--segments 1` to restore the old global behaviour if the
tradeoff is wrong for a given recording.

Division is used rather than subtraction throughout, because illumination drift
is multiplicative and the static walls divide out exactly. The denominator is
`max(bg, 1.0)` to avoid a blow-up where the background is zero (the electrode
clamp reaches 0 and produced ratios of 2e6 before this guard).

## Why geometric NMS was added

deeptangle's latent NMS exists because entangled worms on a dense plate share a
centre of mass, so geometry cannot separate them. Here the worms are sparse and
well separated, and the latent codes - trained on 30-45 px synthetic bodies -
generalise poorly enough to leave **two surviving detections per worm**.
Sweeping `--overlap` from 0.5 to 0.05 moved the count only 10 -> 9, confirming
the duplicates had genuinely distant latent codes.

Geometric NMS (`--nms-radius`, model-space px) suppresses a lower-scoring
detection whose spline hugs a better one. If a recording does get dense and
entangled, lower `--nms-radius` and lean on the latent stage again.

## Centreline accuracy, and why refinement exists

Measured against the segmented medial axis (`scripts/shapecheck.py`), the raw
model splines track the mid-body well but fail at the ends:

| Body region | Model only |
|---|---|
| Mid-body (points 5-43) | 2.61 px |
| Tips (points 0-4, 44-48) | **9.20 px** |

That is exactly why deeptangle's own scripts plot `x[5:-5]` - upstream trims
those five points at each end.

Two things follow.

**The eigenworm basis is not the limit.** Projecting a *measured* centreline
through `weights/eigenworms_transform.npy` reconstructs it to 0.10 model px
(0.4 full-res px). The detector mispredicts coefficients; the basis can express
these postures. So this is fine-tunable (`train.py --load`), not a from-scratch
retrain.

**Scale mattered more than expected.** Sweeping the downscale factor:

| Scale | Body (px) | Length ratio | Coverage@6px |
|---|---|---|---|
| 2.0 | 72 | 0.54 | 0.43 |
| 3.0 | 48 | 0.74 | 0.71 |
| 3.5 | 41 | 0.89 | 0.81 |
| **4.0** | **36** | **0.99** | **0.87** |
| 5.0 | 29 | 1.20 | 0.82 |

The model cannot emit a body longer than its training distribution, so too
little downscaling truncates the spline. 4.0x is the optimum and is the default.

**Tip refinement** then snaps each spline onto the medial axis of the blob it
sits on, isolated inside a corridor around the spline. Worms constantly touch
the wall and debris, and the skeleton of a merged blob wanders (measured length
ratios 1.6-3.6, shifts 17-35 px), so the corridor is essential - but a tight
corridor never sees a tail hook the model missed, so the width escalates
20 -> 36 -> 54 px. The **best** accepted corridor wins, not the first: a tight
corridor that clipped a tail still passes the gates (its axis is simply
shorter), and first-accept then blocks the wider corridor that would have
recovered the missing body.

| | Model only | + refinement |
|---|---|---|
| Refined fraction | - | **91%** |
| Mid-body error | 2.61 px | 1.46 px |
| Coverage@6px | 0.87 | **0.96** |

**Caveat on these numbers.** Where refinement succeeds the centreline *is* the
segmentation's medial axis, and shapecheck compares it against that same medial
axis, so the sub-pixel residual it reports is partly self-referential. What the
numbers legitimately show is **coverage and length** - the spline now spans the
whole body. Accuracy is now bounded by segmentation quality rather than by the
model. Establishing true accuracy needs hand-traced ground truth, which does not
exist yet.

The division of labour after refinement: **deeptangle supplies detection,
identity and tracking through contact; the image supplies the geometry.** The 9%
that fail refinement keep the model spline, the conservative fallback.

Disable with `--no-refine` for pure model output.

## Output

`<stem>_centerlines.csv`, one row per worm per frame:

```
frame, time_s, identity, x0, y0, x1, y1, ... x48, y48
```

Coordinates are full-resolution original-frame pixels. Point order is consistent
within a track but which end is the head is undetermined.

`--video-out` also writes `<stem>_overlay.mp4` with coloured, ID-labelled
centrelines.

## Validation

- `scripts/shapecheck.py` - shape against the medial axis, per-worm table,
  eigenworm residual, 4x zoom crops. **Use this for shape.**
- `scripts/validate.py` - coarse location check and global offset search. Only
  confirms "the spline is inside the body"; worms are ~11 px wide so its
  residual is flat within about +-5 px and it cannot detect shape error.
- `scripts/overlay.py` - annotated segmentation, for tuning blob filters by eye.

## Limitations

- **No absolute head/tail.** Point ordering is consistent *along* a track, but
  which end is the head is undetermined. Planned approach: vote over the
  pre-stimulus baseline (in focus, locomoting, direction of travel reliable) and
  carry the result through the seizure via track identity. Note the synthetic
  training data contains no head/tail cue at all - the renderer's radius profile
  is symmetric about the body midpoint - so a polarity model needs real frames
  or a modified renderer.
- **Refinement inherits segmentation errors.** Where two worms genuinely touch,
  the corridor cannot separate them and the model spline is kept (9% of cases).
  That is the right fallback, but those frames have the tip error above.
- **CPU only.** JAX publishes no CUDA wheels for Windows, so the RTX 3090 is
  unused. Run under WSL2 with `jax[cuda12]` for a large speedup.
- **`--body-px` is global**, default 145 px measured on M9_7. Re-run
  `measure.py` per condition if magnification changed.
- **Seizure postures are out of distribution.** The basis handles them, but the
  detector's learned prior regresses tips toward smooth swimming shapes -
  visible as cut corners on tight terminal hooks. Refinement masks this; a
  fine-tune with a chamber-confined, coil-capable simulator would fix it.
- Frames near the right electrode are trimmed by default and get no detections.
- **The early window is still weak.** After the background fix the bubble false
  positives are gone, but at t~6 s splines are still misplaced on some worms,
  debris is occasionally tracked, and real worms are missed. Contrast is lower
  early in these recordings. Not yet diagnosed.
- **Nothing is validated against real ground truth.** Every accuracy number here
  compares against an automatic segmentation, and where refinement succeeds the
  centreline *is* that segmentation's medial axis. Hand-traced frames are needed
  before any of these numbers should be quoted.

## Layout

```
dtpipe/preprocess.py    domain adaptation: background, ROI, scale, fps
dtpipe/postprocess.py   geometric NMS, tip refinement, flips, track stitching
scripts/measure.py      measure a rig: body length, count, recommended scale
scripts/overlay.py      annotated segmentation, for tuning filters by eye
scripts/smoke.py        scale sweep on one clip, with overlay
scripts/shapecheck.py   shape vs medial axis, eigenworm residual, zoom crops
scripts/validate.py     coarse location check + offset search
scripts/run.py          full pipeline on one video
scripts/batch.py        whole folder
vendor/deeptangle/      upstream repo, one patch (lazy skvideo import)
weights/                released checkpoint
```
