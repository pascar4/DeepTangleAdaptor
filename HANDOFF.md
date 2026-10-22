# Handoff

Everything someone needs to pick this up. Read this before `README.md` — the
README documents *how* the pipeline works; this documents *where it stands*,
*what is wrong with it*, and *what to do next*.

---

## 1. What this is

An adaptation of the released [de(ep)tangle](https://github.com/kirkegaardlab/deeptangle)
checkpoint to C. elegans microfluidics seizure-assay recordings, producing
per-worm centrelines as CSV.

The dataset is 23 recordings of a genotype × drug design — M9 controls, PTZ,
60 V electroshock, isoflurane, and mutants (Unc-25, Dop-3, Cgc-1, N2). The goal
is per-worm body-curvature time series from which seizure phenotypes
(convulsion frequency, time to paralysis) can be computed.

**The checkpoint was never trained on anything like this footage.** Most of the
code here exists to close that gap. The specific mismatches, all measured:

| | Checkpoint expects | These recordings |
|---|---|---|
| Body length | 30–45 px | **~145 px** → downscale 4.0× |
| Frame rate | ~18.2 fps (11 frames / 0.55 s) | 30 fps → keep every 2nd |
| Contrast | worms bright on dark | dark on light → background division, then `1 − ratio` |
| Clutter | random isotropic blobs | wall lines, growing bubbles, debris |

---

## 2. Status: honest version

### Works

- **Detection and per-frame centrelines.** On the one hand-annotated frame:
  **7 TP / 0 FP / 0 FN**, position error 1–10 px (one-to-one Hungarian matching).
- **Background removal.** Per-segment medians handle growing bubbles and the
  static chamber walls. Zero bubble false positives after the fix.
- **Centreline shape**, after medial-axis refinement: 96 % coverage of the body,
  length ratio 0.96, ~91–96 % of splines refined.
- **Runs end to end** on both density regimes without manual intervention.

### Does not work

- **Track identity.** This is the blocking problem. On the isoflurane recording:
  **418 tracks for ~24 worms** — roughly 17 fragments per animal. Median track
  duration 15 s out of 358 s. Detection is stable at 24.4 worms/frame; identity
  is not.
- **Recall in dense scenes.** Roughly 70 % by eye on the isoflurane video —
  never measured properly, because there is no ground truth for dense frames.
  Misses cluster on **coiled worms** (out-of-distribution posture; the simulator
  only produces smooth swimming shapes) and **low-contrast / out-of-focus worms**.
- **Head/tail polarity.** Not implemented. Point ordering is consistent *along* a
  track but which end is the head is undetermined.

### What this means for the science

Usable **now** for per-frame population measures (mean curvature, fraction
paralysed). **Not usable** for per-worm time series — a median 15 s track cannot
support a time-to-paralysis measurement. Fix the linking first.

---

## 3. Setup

```bash
git clone https://github.com/pascar4/DeepTangleAdaptor.git
cd DeepTangleAdaptor

# Python 3.12. JAX has no CUDA wheels on Windows - CPU only unless you use WSL2.
uv venv --python 3.12 .venv
VIRTUAL_ENV=.venv uv pip install -r requirements.txt

# Upstream repo + the two patches in patches/
git clone --depth 1 https://github.com/kirkegaardlab/deeptangle.git vendor/deeptangle
cd vendor/deeptangle && git apply ../../patches/*.diff && cd ../..
VIRTUAL_ENV=.venv uv pip install -e vendor/deeptangle

# Released checkpoint (170 MB)
curl -L https://sid.erda.dk/share_redirect/cEjIpG1yQl -o weights.zip && unzip weights.zip
```

Then, on any new recording — **always measure first**, never assume parameters
carry over:

```bash
.venv/Scripts/python.exe scripts/measure.py "path/to/video.mp4"
.venv/Scripts/python.exe scripts/run.py  "path/to/video.mp4" --out results --video-out
```

Runtime on CPU: ~8 min per 343 s sparse recording, ~70 min per 359 s dense one.
Cost scales with worm count (geometric NMS is O(n²) and refinement is per-spline).

**Data lives on Google Drive**, not in this repo:
`G:\My Drive\C_elegans_video_dataset\Cropped\Main\Videos\` — 23 `.mp4` files.
Nothing in this pipeline writes to Drive.

---

## 4. Work packages

### A — Ground truth (hours, no coding) — DO THIS FIRST

Everything else is unmeasurable without it. There is exactly **one** annotated
frame, from the *sparse* video. The dense regime — where the pipeline actually
struggles — has none.

Tooling already exists:

```bash
.venv/Scripts/python.exe scripts/export_for_annotation.py "video.mp4" --n 12 --out annot
# mark each worm in a distinct saturated colour, save as PNG
.venv/Scripts/python.exe scripts/export_for_annotation.py "video.mp4" --read annot
```

The extractor separates marks by hue peaks (one colour = one worm), which is
what lets crossing worms stay distinguishable.

**Deliverable:** `data/ground_truth/*.csv` for ≥10 frames spanning both videos
and early/mid/late within each.

**Why it matters:** `min_align=0.90` — the parameter separating a duplicate
detection from a genuine crossing — was tuned on a single frame. It may be
overfit. Nobody can tell yet.

### B — Fix track fragmentation (1–2 weeks) — THE REAL PROBLEM

418 tracks for 24 worms. Specific leads, in order:

1. **`directed_distance` has a hardcoded 25 px gate**
   (`vendor/deeptangle/deeptangle/tracking.py:99`). That is *model* space — at
   scale 4.0 it is 100 full-res px. Worse, trackpy does not pass elapsed time to
   the distance function, so a 15-frame `memory` gap gets the same gate as a
   1-frame step. A worm that drifts while occluded cannot relink.
2. **`stitch_tracks`** (`dtpipe/postprocess.py`) is the second-pass repair and
   merged 489 endpoints on the isoflurane run — still not enough. Its
   `max_gap` / `max_jump` were set from sparse-video statistics.
3. **`same-frame dropped 177`** on that run: the co-occurrence guard in
   `stitch_tracks` is not fully transitive, so chained merges still occasionally
   put one identity twice in a frame and a detection is silently discarded. Only
   0.14 %, but it is a real leak.
4. **Detection dropouts** from coiling and defocus break continuity upstream of
   any linking fix.

**Acceptance:** median track duration > 150 s on the isoflurane recording, with
detection recall (measured against package A) not regressing.

### C — Head/tail polarity (~1 week)

Designed but not built. Approach is in `README.md`. The core insight:

> Polarity is a **track** property, not a frame property. Establish it during the
> pre-stimulus baseline — worms in focus, locomoting, direction of travel
> reliable — then carry it through the seizure via track identity.

**Blocker worth knowing before starting:** the synthetic training data contains
**no head/tail cue at all**. The renderer's radius profile
(`celegans/clips.py:27`) is symmetric about the body midpoint. A polarity model
must train on real frames or a modified renderer. Naively training on stock
synthetic data will latch onto a 0.14 px sampling artifact at index 0 and
transfer at chance.

Depends on B (needs stable tracks) and A (needs validation).

---

## 5. Design decisions and why

Non-obvious choices. Changing any of these without reading the reason will
re-introduce a bug that has already been found and fixed once.

**Downscale 4.0×, not 3.5×.** Measured by sweeping and scoring spline length
against the segmented medial axis:

| Scale | Body px | Length ratio | Coverage@6px |
|---|---|---|---|
| 2.0 | 72 | 0.54 | 0.43 |
| 3.5 | 41 | 0.89 | 0.81 |
| **4.0** | **36** | **0.99** | **0.87** |
| 5.0 | 29 | 1.20 | 0.82 |

The model cannot emit a body longer than its training distribution, so *too
little* downscaling truncates the spline. Counter-intuitive — bigger is worse.

**Background division, not subtraction.** Theoretically right (transmitted light
attenuates multiplicatively), but measured on this rig the two agree to **98 %
IoU** — the illumination is flat (background 137 grey; left half 138, right half
135). Either works here. Keep division as insurance for a rig with a real
gradient.

**Per-segment backgrounds, not global, not rolling.**
- Global holds whatever bubble configuration dominates the recording; frames
  that differ get huge residuals. Measured: **18.8 %** of early detections were
  on bubble rims.
- Rolling would absorb a *paralysed* worm into the background — erasing exactly
  the phenotype being measured. This is a seizure assay.
- Per-segment medians split the difference. Result: 18.8 % → **0.0 %**.

**The instability mask is OFF.** It was added alongside per-segment backgrounds
to exclude non-static pixels. Testing showed per-segment backgrounds fix the
bubbles *on their own*, and the mask additionally **deletes a real worm** near a
bubble margin (confirmed against the hand annotation — the missed worm at
(256,111) is the one it excludes). Re-enable with `--instability-mask` only with
a measured reason.

**Geometric NMS needs tangent alignment, not just overlap.** Two detections of
one worm are offset *along* the body by 40–60 px, so mean curve distance does not
separate them from a neighbour. Overlap fraction does. But two *crossing* worms
also overlap — transversally. Requiring `min_align ≥ 0.90` between tangents keeps
them apart. Without it, the annotated crossing pair collapsed into one track.

**The eigenworm basis is NOT the limit.** Projecting a *measured* centreline
through `weights/eigenworms_transform.npy` reconstructs it to **0.10 model px
(0.4 full-res px)**. The detector mispredicts coefficients; the 12-component
basis can express these postures. **This means fine-tuning (`train.py --load`) is
viable — a from-scratch retrain is not required.** Important: it rules out the
most expensive possible fix.

**Tip refinement, and its honest limits.** Raw model splines track the mid-body
to 2.61 px but the outermost 5 points at each end to 9.20 px — which is why
deeptangle's own scripts plot `x[5:-5]`. Refinement snaps each spline onto the
segmentation's medial axis inside a corridor that widens 20 → 36 → 54 px, taking
the **best** accepted corridor rather than the first (a tight corridor that
clipped a tail still passes the gates and would block the wider one).

> **Caveat that must not be lost:** where refinement succeeds, the centreline
> *is* the segmentation's medial axis — and `shapecheck.py` scores it against
> that same medial axis. The sub-pixel residuals it reports are partly
> self-referential. What they legitimately show is **coverage and length**.
> Accuracy is now bounded by segmentation quality, not by the model. Only
> package A gives a real accuracy number.

---

## 6. Upstream bugs found

Both are genuine defects in `kirkegaardlab/deeptangle` (upstream commit
`7c9775d`), patched in `patches/`. **Worth filing as issues** — the first will hit
anyone who runs their tracker on a full-length recording.

**1. `identity_assignment` crashes on a frame with zero detections**
(`deeptangle/tracking.py`). `pred.w.reshape(N, -1)` cannot infer `-1` from an
empty array:

```
ValueError: cannot reshape array of size 0 into shape (0,newaxis)
```

Short segments never hit it; a 343 s recording does. Fix: reshape to an explicit
product.

**2. `deeptangle/dataset/io.py` imports `skvideo` at module scope**, so importing
the package at all requires an unmaintained library that breaks on numpy ≥ 2.
`video_to_clips` does not need it. Fix: make the import lazy.

Non-bugs worth reporting as documentation gaps:
- The checkpoint's effective frame rate (~18.2 fps) is implicit in
  `clip_duration / nframes` and is not documented anywhere; it matters for
  anyone with different footage.
- `--num_batches` in `examples/track.py` is described as "batches to do
  simultaneously" but is the number of *sequential* chunks.
- `FLAGS.merge` in `train.py` is defined and never used.

---

## 7. Sharp edges

- **Constants that must agree across files but are not shared.** `ls_dim = 8` in
  `tracking.py:98` is the latent dimension — change `--latent_dim` and tracking
  silently slices the wrong columns. `X[i][j][25]` assumes 49 skeleton points.
- **`--body-px` defaults to 145**, measured on two recordings from different
  sessions that happened to match. Re-run `measure.py` per condition.
- **CPU only on Windows.** JAX publishes no CUDA wheels. There is an RTX 3090 on
  the machine sitting idle; WSL2 + `jax[cuda12]` would be a large speedup and is
  probably the single cheapest performance win.
- **No tests.** Nothing guards the NMS or linking against regression.
- `results_full/` for the M9_7 recording was never produced — the full run
  crashed on the empty-frame bug before it was patched. Re-run it.

---

## 8. Script map

| Script | Purpose |
|---|---|
| `scripts/measure.py` | Measure a rig: body length, worm count, recommended scale and stride. **Run on every new recording.** |
| `scripts/run.py` | Full pipeline on one video |
| `scripts/batch.py` | Whole folder |
| `scripts/bgdemo.py` | Background removal step by step: global vs per-segment, division vs subtraction, thresholded. Best debugging tool here. |
| `scripts/shapecheck.py` | Centreline shape vs medial axis, eigenworm residual, 4× zoom crops. **Use this for shape, not `validate.py`.** |
| `scripts/validate.py` | Coarse location check. Cannot detect shape error — worms are ~11 px wide so its residual is flat within ±5 px. |
| `scripts/overlay.py` | Annotated segmentation, for tuning blob filters by eye |
| `scripts/smoke.py` | Scale sweep on one clip |
| `scripts/export_for_annotation.py` | Export frames for hand annotation and read them back |

`dtpipe/preprocess.py` — domain adaptation (background, ROI, scale, fps).
`dtpipe/postprocess.py` — geometric NMS, tip refinement, flip canonicalisation,
track stitching.
