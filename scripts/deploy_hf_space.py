"""Stage and push Sage to a Hugging Face Space (Docker SDK).

HF Docker Spaces build the image server-side from a Dockerfile committed to
the Space's own git repo -- so this script pushes *source*, not a pre-built
image or a pre-built frontend/dist/. The Space's Dockerfile itself runs
`npm ci && npm run build` in its frontend-build stage (see the root
Dockerfile), same as a local `docker build .` would.

Usage:
    # Dry run (default) -- prints what would be uploaded, touches no network.
    .venv/bin/python scripts/deploy_hf_space.py --repo-id <user>/<space>

    # Actually push to the Space repo on huggingface.co. Requires
    # HF_TOKEN in the environment (or a prior `huggingface-cli login`).
    .venv/bin/python scripts/deploy_hf_space.py --repo-id <user>/<space> --push

This pushes to a shared, public Hugging Face repo and (re)deploys the live
Space -- get explicit confirmation before running with --push for real. See
deploy/huggingface/DEPLOY.md for the full one-time setup and verification
checklist.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Everything the Space's Dockerfile needs to build + run, and nothing else
# (no tests/, docs/, .venv/, local data/db). frontend/ is staged as source
# (minus node_modules/dist, rebuilt by the Space itself), not frontend/dist.
FILES_TO_STAGE = [
    "Dockerfile",
    ".dockerignore",
    "pyproject.toml",
]
DIRS_TO_STAGE = [
    "api",
    "sage",
    "config",
    "frontend",
    "deploy/huggingface/prebuilt",
]
# Never stage these even if present under a staged directory.
EXCLUDE_DIR_NAMES = {"__pycache__", "node_modules", "dist", ".git"}
EXCLUDE_SUFFIXES = {".pyc"}

# deploy/huggingface/README.md holds the Space's required YAML frontmatter
# (sdk: docker, app_port, title, ...) -- HF reads README.md at the Space
# repo root for this, so it's staged there, not at its source path.
SPACE_README_SRC = REPO_ROOT / "deploy" / "huggingface" / "README.md"


def _ignore(_dir: str, names: list[str]) -> set[str]:
    return {n for n in names if n in EXCLUDE_DIR_NAMES or Path(n).suffix in EXCLUDE_SUFFIXES}


def stage(staging_dir: Path) -> list[Path]:
    """Copy everything the Space needs into staging_dir. Returns the list of
    top-level paths staged, for the dry-run summary."""
    staged: list[Path] = []

    for rel in FILES_TO_STAGE:
        src = REPO_ROOT / rel
        if not src.is_file():
            raise FileNotFoundError(f"expected file missing: {src}")
        dest = staging_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        staged.append(src)

    for rel in DIRS_TO_STAGE:
        src = REPO_ROOT / rel
        if not src.is_dir():
            raise FileNotFoundError(f"expected directory missing: {src}")
        # Stage under the directory's own top-level name (e.g.
        # "deploy/huggingface/prebuilt" -> staging_dir/"prebuilt"), matching
        # where the Dockerfile's COPY instructions expect to find it relative
        # to the build context root for everything except the nested
        # deploy/huggingface/prebuilt path, which the Dockerfile references
        # by its full relative path -- so that one is staged at the same
        # relative path instead of flattened.
        dest = staging_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dest, ignore=_ignore, dirs_exist_ok=True)
        staged.append(src)

    if not SPACE_README_SRC.is_file():
        raise FileNotFoundError(f"expected Space README missing: {SPACE_README_SRC}")
    shutil.copy2(SPACE_README_SRC, staging_dir / "README.md")
    staged.append(SPACE_README_SRC)

    return staged


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Target Space, e.g. 'yourname/sage'.",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Actually upload to the Hub. Without this flag, only stages "
        "locally and prints what would be uploaded.",
    )
    parser.add_argument(
        "--commit-message",
        default="Deploy Sage",
        help="Commit message for the Space repo update.",
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="sage-hf-space-") as tmp:
        staging_dir = Path(tmp)
        staged = stage(staging_dir)

        print(f"Staged {len(staged)} top-level paths into {staging_dir}:")
        for p in sorted(staged):
            rel = p.relative_to(REPO_ROOT) if p.is_relative_to(REPO_ROOT) else p
            print(f"  - {rel}")

        if not args.push:
            print(
                "\nDry run only (pass --push to actually upload). "
                f"Would push to Space repo '{args.repo_id}'."
            )
            return 0

        # Imported lazily so a plain dry-run (the common/safe case) never
        # requires huggingface_hub to be importable or HF_TOKEN to be set.
        from huggingface_hub import HfApi

        print(f"\nPushing to Space '{args.repo_id}' ...")
        api = HfApi()
        api.upload_folder(
            folder_path=str(staging_dir),
            repo_id=args.repo_id,
            repo_type="space",
            commit_message=args.commit_message,
        )
        print("Done. Check the Space's Logs tab for build progress.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
