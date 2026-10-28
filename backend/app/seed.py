"""Seeds the shared ('system') corpora so every agent is demoable on first boot.

The Code agent indexes this very repository - so you can ask the app how it works.
"""

from pathlib import Path

from app.container import Container
from app.core.ingestion.pipeline import SYSTEM_OWNER
from app.core.observability.logging import get_logger

log = get_logger(__name__)

_TEXT_SUFFIXES = {".md", ".txt", ".pdf", ".docx", ".vtt"}

# What the Code agent indexes about itself: source, tests, infra and docs - not sample data.
CODE_PATHS = [
    "backend/app",
    "backend/tests",
    "backend/evals/run.py",
    "backend/scripts",
    "backend/pyproject.toml",
    "frontend/src",
    "frontend/package.json",
    "docs",
    "README.md",
    "Dockerfile",
    ".github/workflows",
    "CLAUDE.md",
]


def _files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix in _TEXT_SUFFIXES)


def seed(container: Container, repo_root: Path, force: bool = False) -> None:
    kb = container.kb
    samples = container.settings.samples_dir

    def needed(agent: str) -> bool:
        return force or kb.retriever(agent).is_empty()

    container.status["seed"] = "running"
    if needed("docs"):
        for path in _files(samples / "docs"):
            kb.ingest_file("docs", path.name, path.read_bytes(), SYSTEM_OWNER)
    if needed("meeting"):
        for path in _files(samples / "meetings"):
            kb.ingest_file("meeting", path.name, path.read_bytes(), SYSTEM_OWNER)
    if needed("career"):
        for path in _files(samples / "career"):
            role = "resume" if path.name.startswith("resume") else "jd"
            kb.ingest_file("career", path.name, path.read_bytes(), SYSTEM_OWNER, {"doc_role": role})
    if needed("code"):
        kb.ingest_directory("code", repo_root, SYSTEM_OWNER, repo="atlas", include=CODE_PATHS)
    container.status["seed"] = "done"
    log.info("seed_complete")


def main() -> None:
    """CLI: re-index after the codebase changes, e.g. `python -m app.seed --agent code`."""
    import argparse

    from app.container import build_container
    from app.core.config import get_settings

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent", choices=["docs", "code", "meeting", "career", "all"], default="all"
    )
    args = parser.parse_args()
    container = build_container(get_settings())
    repo_root = Path(__file__).resolve().parents[2]
    if args.agent in ("code", "all"):
        report = container.kb.ingest_directory(
            "code", repo_root, SYSTEM_OWNER, repo="atlas", include=CODE_PATHS
        )
        print(f"code: {report.documents} files, {report.chunks} chunks")
    if args.agent != "code":
        seed(container, repo_root, force=True)


if __name__ == "__main__":
    main()
