"""Pre-flight checks that fail with an explanation rather than a stack trace."""
from pathlib import Path

WEIGHTS_FILES = ["arrays.npy", "tree.pkl", "eigenworms_transform.npy",
                 "experiment.json"]


def require_weights(path):
    """Raise a message a newcomer can act on if the checkpoint is absent.

    The checkpoint is 164 MB and belongs to the de(ep)tangle authors, so it is
    deliberately not in this repository. Without this check the failure is
    `FileNotFoundError: weights/experiment.json`, which says nothing about what
    to do.
    """
    p = Path(path)
    missing = [f for f in WEIGHTS_FILES if not (p / f).exists()]
    if not missing:
        return

    if not p.exists():
        what = f"The weights directory '{p}' does not exist."
    else:
        what = f"'{p}' is missing: {', '.join(missing)}"

    raise SystemExit(
        "\n"
        "-------------------------------------------------------------\n"
        f"{what}\n"
        "\n"
        "The de(ep)tangle checkpoint (164 MB) is not included in this\n"
        "repository - it belongs to the upstream authors and is fetched\n"
        "separately.\n"
        "\n"
        "Run the setup script from the repository root:\n"
        "\n"
        "    python bootstrap.py\n"
        "\n"
        "It downloads and unpacks the checkpoint, clones upstream\n"
        "de(ep)tangle, applies the required patches, and checks your\n"
        "environment. It is safe to re-run.\n"
        "-------------------------------------------------------------"
    )
