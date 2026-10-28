"""Offline evaluation reports (admin only)."""

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter

from app.api.deps import PrincipalDep, ensure_action

router = APIRouter(prefix="/api/evals", tags=["evals"])

REPORTS_DIR = Path(__file__).resolve().parents[2] / "evals" / "reports"


@router.get("/reports")
def list_reports(principal: PrincipalDep) -> list[dict[str, Any]]:
    ensure_action(principal, "view_eval_reports")
    reports = []
    for path in sorted(REPORTS_DIR.glob("*.json"), reverse=True):
        try:
            reports.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    return reports
