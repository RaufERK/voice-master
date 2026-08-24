from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def batch_path(work_dir: Path, batch_index: int) -> Path:
    return work_dir / "batches" / f"{batch_index:04d}.json"


def tts_path(work_dir: Path, segment_id: int) -> Path:
    return work_dir / "tts" / f"{segment_id:04d}.mp3"


def load_done_translations(work_dir: Path) -> dict[int, str]:
    batches_dir = work_dir / "batches"
    if not batches_dir.exists():
        return {}
    translations: dict[int, str] = {}
    for path in sorted(batches_dir.glob("*.json")):
        payload = load_json(path) or {}
        if not payload.get("ok"):
            continue
        for item in payload.get("translated") or []:
            translations[int(item["id"])] = str(item["text"])
    return translations
