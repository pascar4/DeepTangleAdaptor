"""
Run the pipeline across a folder of recordings.

Usage:
  python scripts/batch.py "G:/My Drive/.../Videos" --out results --workers 1
"""
import argparse
import glob
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--out", default="results")
    ap.add_argument("--glob", default="*.mp4")
    ap.add_argument("--body-px", type=float, default=145.0)
    ap.add_argument("--duration", type=float, default=0.0)
    ap.add_argument("--video-out", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    a = ap.parse_args()

    vids = sorted(glob.glob(os.path.join(a.folder, a.glob)))
    print(f"{len(vids)} videos in {a.folder}\n")
    os.makedirs(a.out, exist_ok=True)

    t_all = time.time()
    failures = []
    for i, v in enumerate(vids, 1):
        stem = os.path.splitext(os.path.basename(v))[0]
        csv = os.path.join(a.out, f"{stem}_centerlines.csv")
        if a.skip_existing and os.path.exists(csv):
            print(f"[{i}/{len(vids)}] skip (exists)  {stem}")
            continue
        print(f"[{i}/{len(vids)}] {stem}")
        cmd = [PY, os.path.join(HERE, "run.py"), v,
               "--out", a.out, "--body-px", str(a.body_px)]
        if a.duration:
            cmd += ["--duration", str(a.duration)]
        if a.video_out:
            cmd += ["--video-out"]
        t0 = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            failures.append(stem)
            print(f"    FAILED: {r.stderr.strip().splitlines()[-1:]}")
        else:
            tail = [l for l in r.stdout.splitlines()
                    if "tracks," in l or "rows)" in l]
            for l in tail:
                print("   " + l.strip())
            print(f"    {time.time()-t0:.0f}s")

    print(f"\ntotal {time.time()-t_all:.0f}s, {len(failures)} failed")
    for f in failures:
        print("  !", f)


if __name__ == "__main__":
    main()
