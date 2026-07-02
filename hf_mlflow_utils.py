"""Portable MLflow path helpers for local development and Docker."""

from __future__ import annotations

import sqlite3
from pathlib import Path

MLRUNS_MARKER = "/mlruns/"


def resolve_artifact_uri(uri: str, app_root: Path) -> str:
    """Map any absolute mlruns path to the current application root."""
    if not uri:
        return uri
    idx = uri.find(MLRUNS_MARKER)
    if idx == -1:
        return uri
    return str(app_root.resolve()) + uri[idx:]


def normalize_mlflow_paths(app_root: Path) -> None:
    """Rewrite absolute host paths in mlflow.db and MLmodel metadata files."""
    app_root = app_root.resolve()
    db_path = app_root / "mlflow.db"
    if not db_path.exists():
        return

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    for table, column in (
        ("experiments", "artifact_location"),
        ("runs", "artifact_uri"),
        ("logged_models", "artifact_location"),
    ):
        try:
            cur.execute(f"SELECT rowid, {column} FROM {table}")
        except sqlite3.OperationalError:
            continue
        for rowid, uri in cur.fetchall():
            new_uri = resolve_artifact_uri(uri, app_root)
            if new_uri != uri:
                cur.execute(
                    f"UPDATE {table} SET {column} = ? WHERE rowid = ?",
                    (new_uri, rowid),
                )
    conn.commit()
    conn.close()

    mlruns = app_root / "mlruns"
    if not mlruns.exists():
        return

    for mlmodel_path in mlruns.rglob("MLmodel"):
        text = mlmodel_path.read_text(encoding="utf-8")
        lines = text.splitlines()
        if not lines or not lines[0].startswith("artifact_path:"):
            continue
        current = lines[0].split(":", 1)[1].strip()
        updated = resolve_artifact_uri(current, app_root)
        if updated != current:
            lines[0] = f"artifact_path: {updated}"
            mlmodel_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
