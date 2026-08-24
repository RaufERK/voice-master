from __future__ import annotations

from pathlib import Path

import yaml

from . import Glossary


def load_glossary(path: Path | None) -> Glossary | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"Glossary not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return Glossary(
        series=str(data.get("series") or ""),
        context=str(data.get("context") or "").strip(),
        names={str(key): str(value) for key, value in (data.get("names") or {}).items()},
        terms={str(key): str(value) for key, value in (data.get("terms") or {}).items()},
        notes=[str(item) for item in (data.get("notes") or [])],
    )


def glossary_prompt_block(glossary: Glossary | None) -> str:
    if glossary is None:
        return ""
    lines: list[str] = []
    if glossary.series:
        lines.append(f"Lecture: {glossary.series}")
    if glossary.context:
        lines.append(glossary.context.strip())
    if glossary.names:
        lines.append("Proper names:")
        for source, target in glossary.names.items():
            lines.append(f"- {source} → {target}")
    if glossary.terms:
        lines.append("Terms:")
        for source, target in glossary.terms.items():
            lines.append(f"- {source} → {target}")
    if glossary.notes:
        lines.append("Notes:")
        for note in glossary.notes:
            lines.append(f"- {note}")
    return "\n".join(lines)
