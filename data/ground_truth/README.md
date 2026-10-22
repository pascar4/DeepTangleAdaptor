# Ground truth

## frame_000205_marks.csv

Hand annotation by Paul of frame 205 (t=6.83 s) of
`2023-08-25 13-00-51 Pr 7 Microfluidics M9_7.mp4`. Seven worms, each marked in a
distinct colour on the background-removed export; extracted with
`scripts/export_for_annotation.py --read-image`.

Coordinates are full-resolution original-frame pixels (mark centroid, not a
centreline). `arc_px` is the drawn stroke length, `traced` whether the mark was
a line rather than a blob.

**Provenance caveat:** these values are transcribed from the extraction run log.
The original CSV lived in a session temp directory that has since been cleared.
The numbers are exactly as the extractor printed them, but the file itself is a
reconstruction. Re-export from the annotated image to regenerate it canonically.

## What this one frame established

Measured with one-to-one Hungarian matching (`scale=4.0`, `min_align=0.90`,
instability mask OFF):

| | |
|---|---|
| True positives | 7 |
| False positives | 0 |
| False negatives | 0 |
| Centreline position error | 1-10 px |

It also settled three design questions:
- the instability mask deletes a real worm near a bubble -> left OFF by default
- geometric NMS needs tangent alignment, or it merges a crossing pair
- positional accuracy is genuinely good (this is the only non-self-referential
  accuracy number in the project)

## What is still missing

**This is one frame, from the SPARSE video.** There is no ground truth at all for
the dense regime (the isoflurane recording, ~25-30 worms), which is where the
pipeline actually struggles. `min_align=0.90` was tuned on this single frame and
may be overfit. See work package A in HANDOFF.md.
