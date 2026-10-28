"""Push this repository to a Hugging Face Docker Space.

    export HF_TOKEN=hf_xxx HF_SPACE=<user>/<space>
    uv run --with huggingface_hub python scripts/deploy_hf.py

The Space needs a README with YAML front-matter (sdk: docker). The project README is for
GitHub reviewers, so the front-matter is prepended only in the uploaded copy.
Secrets (ANTHROPIC_API_KEY, HF_TOKEN, JWT_SECRET, ADMIN_PASSWORD, STAFF_PASSWORD) are set
once in the Space settings - they are never uploaded from here.
"""

import os
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[2]
FRONT_MATTER = """---
title: Atlas Conversational AI
emoji: 🧭
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
short_description: One RAG core, four LangGraph agents (docs, code, meetings, careers)
---

"""
IGNORE = [
    ".git/*",
    "**/node_modules/*",
    "**/.venv/*",
    "**/__pycache__/*",
    "**/.angular/*",
    "frontend/dist/*",
    "backend/static/*",
    "backend/data/*",
    "**/.env",
    "docs/screenshots/*",
    "**/.pytest_cache/*",
    "**/.mypy_cache/*",
    "**/.ruff_cache/*",
]


def main() -> int:
    token, space = os.environ.get("HF_TOKEN"), os.environ.get("HF_SPACE")
    if not token or not space:
        print("Set HF_TOKEN and HF_SPACE=<user>/<space>", file=sys.stderr)
        return 1
    api = HfApi(token=token)
    api.create_repo(space, repo_type="space", space_sdk="docker", exist_ok=True)
    api.upload_folder(
        repo_id=space,
        repo_type="space",
        folder_path=ROOT,
        ignore_patterns=[*IGNORE, "README.md"],
        commit_message="Deploy Atlas",
    )
    with tempfile.TemporaryDirectory() as tmp:
        readme = Path(tmp) / "README.md"
        readme.write_text(
            FRONT_MATTER + (ROOT / "README.md").read_text(encoding="utf-8"), encoding="utf-8"
        )
        api.upload_file(
            path_or_fileobj=readme,
            path_in_repo="README.md",
            repo_id=space,
            repo_type="space",
            commit_message="Space README",
        )
    print(f"Deployed: https://huggingface.co/spaces/{space}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
