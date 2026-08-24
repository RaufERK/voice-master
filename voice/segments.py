from __future__ import annotations

from . import Cue, Segment


def merge_cues(cues: list[Cue], *, gap_seconds: float) -> list[Segment]:
    if not cues:
        return []
    segments: list[Segment] = []
    current = [cues[0]]
    for cue in cues[1:]:
        previous = current[-1]
        if cue.start - previous.end <= gap_seconds:
            current.append(cue)
            continue
        segments.append(_to_segment(len(segments) + 1, current))
        current = [cue]
    segments.append(_to_segment(len(segments) + 1, current))
    return segments


def filter_window(cues: list[Cue], start: float, end: float) -> list[Cue]:
    selected: list[Cue] = []
    for cue in cues:
        if cue.end <= start or cue.start >= end:
            continue
        if _is_junk(cue.text):
            continue
        selected.append(cue)
    return selected


def _is_junk(text: str) -> bool:
    words = text.split()
    return len(words) < 3 and len(text) < 16


def _to_segment(segment_id: int, cues: list[Cue]) -> Segment:
    text = " ".join(cue.text for cue in cues)
    return Segment(
        segment_id=segment_id,
        start=cues[0].start,
        end=cues[-1].end,
        text=text,
        cue_ids=tuple(cue.cue_id for cue in cues),
    )
