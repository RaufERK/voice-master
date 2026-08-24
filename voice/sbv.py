from __future__ import annotations

import re
from pathlib import Path

from . import Cue

_TS = re.compile(
    r"^(?P<h>\d+):(?P<m>\d{2}):(?P<s>\d{2})\.(?P<ms>\d+)"
    r",(?P<h2>\d+):(?P<m2>\d{2}):(?P<s2>\d{2})\.(?P<ms2>\d+)\s*$"
)


def parse_sbv(text: str) -> list[Cue]:
    cues: list[Cue] = []
    blocks = re.split(r"\n\s*\n", text.strip())
    cue_id = 1
    for block in blocks:
        lines = [line.rstrip() for line in block.splitlines() if line.strip()]
        if len(lines) < 2:
            continue
        match = _TS.match(lines[0])
        if match is None:
            continue
        body = _clean_text(" ".join(lines[1:]))
        if not body:
            continue
        cues.append(
            Cue(
                cue_id=cue_id,
                start=_timestamp_to_seconds(match, "h", "m", "s", "ms"),
                end=_timestamp_to_seconds(match, "h2", "m2", "s2", "ms2"),
                text=body,
            )
        )
        cue_id += 1
    return cues


def load_sbv(path: Path) -> list[Cue]:
    return parse_sbv(path.read_text(encoding="utf-8"))


def _timestamp_to_seconds(match: re.Match[str], h: str, m: str, s: str, ms: str) -> float:
    millis = match.group(ms).ljust(3, "0")[:3]
    return (
        int(match.group(h)) * 3600
        + int(match.group(m)) * 60
        + int(match.group(s))
        + int(millis) / 1000
    )


def _clean_text(text: str) -> str:
    cleaned = text.replace("\xa0", " ").replace("\u200b", "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned
