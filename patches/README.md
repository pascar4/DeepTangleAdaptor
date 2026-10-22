# Patches to upstream de(ep)tangle

Applied against `kirkegaardlab/deeptangle` at commit `7c9775d` (see
`UPSTREAM_COMMIT`). Both are genuine upstream defects, not local workarounds —
see HANDOFF.md section 6, and consider filing them as issues.

```bash
git clone --depth 1 https://github.com/kirkegaardlab/deeptangle.git vendor/deeptangle
cd vendor/deeptangle && git apply ../../patches/*.diff
```

| Patch | Fixes |
|---|---|
| `01-lazy-skvideo-import.diff` | `dataset/io.py` imports `skvideo` at module scope, so importing the package needs an unmaintained library that breaks on numpy >= 2. `video_to_clips` does not use it. |
| `02-empty-frame-crash.diff` | `identity_assignment` crashes on a frame with zero detections: `reshape(0, -1)` cannot infer the free dimension from an empty array. Hits any full-length recording. |
