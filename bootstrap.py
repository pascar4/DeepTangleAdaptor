"""
One-command setup. Run this before anything else:

    python bootstrap.py

Does the three things that are not in the repo and cannot be:

  1. downloads the released de(ep)tangle checkpoint (164 MB)
  2. clones upstream de(ep)tangle and applies the two patches in patches/
  3. checks the Python environment

Pure stdlib - no curl, wget or unzip needed, so it works the same on Windows,
macOS and Linux. Safe to re-run; anything already in place is skipped.
"""
from __future__ import annotations

import io
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "weights"
VENDOR = ROOT / "vendor" / "deeptangle"
PATCHES = ROOT / "patches"

WEIGHTS_URL = "https://sid.erda.dk/share_redirect/cEjIpG1yQl"
WEIGHTS_FILES = ["arrays.npy", "tree.pkl", "eigenworms_transform.npy",
                 "experiment.json"]
UPSTREAM = "https://github.com/kirkegaardlab/deeptangle.git"


def ok(msg):
    print(f"  [ok]   {msg}")


def info(msg):
    print(f"  ...    {msg}")


def fail(msg):
    print(f"  [FAIL] {msg}")


def step(n, msg):
    print(f"\n{n}. {msg}")


# ---------------------------------------------------------------- weights
def weights_present():
    return WEIGHTS.is_dir() and all((WEIGHTS / f).exists() for f in WEIGHTS_FILES)


def download_weights():
    step(1, "de(ep)tangle checkpoint")
    if weights_present():
        ok(f"already present in {WEIGHTS}")
        return True

    if WEIGHTS.is_dir():
        missing = [f for f in WEIGHTS_FILES if not (WEIGHTS / f).exists()]
        fail(f"{WEIGHTS} exists but is incomplete, missing: {missing}")
        info("delete the folder and re-run to fetch a clean copy")
        return False

    info(f"downloading 164 MB from {WEIGHTS_URL}")
    info("(this is the published checkpoint from the de(ep)tangle authors;")
    info(" it is not redistributed in this repo)")
    buf = io.BytesIO()
    try:
        with urllib.request.urlopen(WEIGHTS_URL, timeout=120) as r:
            total = int(r.headers.get("Content-Length", 0))
            got = 0
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                buf.write(chunk)
                got += len(chunk)
                if total:
                    pct = 100 * got / total
                    print(f"\r         {got/1e6:6.1f} / {total/1e6:.1f} MB "
                          f"({pct:5.1f}%)", end="", flush=True)
            print()
    except Exception as e:
        fail(f"download failed: {type(e).__name__}: {e}")
        info("if you are behind a proxy or the host is down, fetch it manually:")
        info(f"  {WEIGHTS_URL}")
        info(f"  then unzip so that {WEIGHTS / 'experiment.json'} exists")
        return False

    buf.seek(0)
    try:
        with zipfile.ZipFile(buf) as z:
            z.extractall(ROOT)
    except zipfile.BadZipFile:
        fail("downloaded file is not a valid zip - the link may have changed")
        return False

    if weights_present():
        ok(f"installed into {WEIGHTS}")
        return True
    fail(f"extracted, but {WEIGHTS} is still missing expected files")
    return False


# ----------------------------------------------------------------- vendor
def setup_vendor():
    step(2, "upstream de(ep)tangle + patches")
    if (VENDOR / "deeptangle" / "model.py").exists():
        ok(f"already cloned at {VENDOR}")
    else:
        if not shutil.which("git"):
            fail("git not found on PATH - install git and re-run")
            return False
        VENDOR.parent.mkdir(parents=True, exist_ok=True)
        info("cloning upstream repository")
        r = subprocess.run(["git", "clone", "--depth", "1", UPSTREAM, str(VENDOR)],
                           capture_output=True, text=True)
        if r.returncode:
            fail(f"clone failed: {r.stderr.strip().splitlines()[-1:]}")
            return False
        ok("cloned")

    diffs = sorted(PATCHES.glob("*.diff"))
    if not diffs:
        fail(f"no .diff files in {PATCHES}")
        return False

    # already applied?  `git apply --reverse --check` succeeds if so
    applied = subprocess.run(
        ["git", "apply", "--reverse", "--check"] + [str(d) for d in diffs],
        cwd=VENDOR, capture_output=True, text=True)
    if applied.returncode == 0:
        ok(f"{len(diffs)} patches already applied")
        return True

    r = subprocess.run(["git", "apply"] + [str(d) for d in diffs],
                       cwd=VENDOR, capture_output=True, text=True)
    if r.returncode:
        fail(f"could not apply patches: {r.stderr.strip()}")
        info("the upstream default branch may have moved past "
             f"{(PATCHES / 'UPSTREAM_COMMIT').read_text().strip()}")
        return False
    ok(f"applied {len(diffs)} patches "
       "(empty-frame crash, lazy skvideo import)")
    return True


# -------------------------------------------------------------- environment
def check_env():
    step(3, "Python environment")
    v = sys.version_info
    print(f"  ...    python {v.major}.{v.minor}.{v.micro} at {sys.executable}")
    if (v.major, v.minor) < (3, 10) or (v.major, v.minor) > (3, 13):
        fail(f"python {v.major}.{v.minor} is outside the tested 3.10-3.13 range "
             "(3.12 recommended; JAX has no 3.14 wheels yet)")

    missing = []
    for mod, label in [("jax", "jax"), ("haiku", "dm-haiku"), ("cv2", "opencv"),
                       ("numba", "numba"), ("trackpy", "trackpy"),
                       ("skimage", "scikit-image"), ("pandas", "pandas")]:
        try:
            __import__(mod)
        except ImportError:
            missing.append(label)
    if missing:
        fail(f"missing packages: {', '.join(missing)}")
        info("  pip install -r requirements.txt")
        return False
    ok("all required packages importable")

    try:
        import deeptangle  # noqa: F401
        ok("deeptangle importable")
    except ImportError:
        fail("deeptangle not installed")
        info("  pip install -e vendor/deeptangle")
        return False

    try:
        import jax
        devs = jax.devices()
        ok(f"jax devices: {devs}")
        if all(d.platform == "cpu" for d in devs):
            info("CPU only. On Windows this is expected - JAX publishes no")
            info("CUDA wheels for Windows. Use WSL2 + jax[cuda12] for GPU.")
    except Exception as e:
        fail(f"jax failed to initialise: {e}")
        return False
    return True


def main():
    print("de(ep)tangle adaptor - setup")
    print(f"repo: {ROOT}")
    results = [download_weights(), setup_vendor(), check_env()]

    print("\n" + "-" * 60)
    if all(results):
        py = (".venv\\Scripts\\python.exe" if os.name == "nt"
              else ".venv/bin/python")
        print("Setup complete. Next:\n")
        print(f'  {py} scripts/measure.py "path/to/video.mp4"')
        print(f'  {py} scripts/run.py     "path/to/video.mp4" --out results '
              "--video-out\n")
        print("Always run measure.py on a new recording first - the downscale")
        print("factor depends on worm body length in pixels and must not be")
        print("assumed to carry over between rigs. See HANDOFF.md.")
    else:
        print("Setup incomplete - see [FAIL] lines above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
