from __future__ import annotations

from . import (
    DEFAULT_MAX_SPEECH_CHARS,
    DEFAULT_MAX_SPEECH_SECONDS,
    Cue,
    Segment,
)


def merge_cues(
    cues: list[Cue],
    *,
    gap_seconds: float,
    max_chars: int = DEFAULT_MAX_SPEECH_CHARS,
    max_seconds: float = DEFAULT_MAX_SPEECH_SECONDS,
) -> list[Segment]:
    if not cues:
        return []
    segments: list[Segment] = []
    current = [cues[0]]
    for cue in cues[1:]:
        if _can_join_speech(current, cue, gap_seconds=gap_seconds, max_chars=max_chars, max_seconds=max_seconds):
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


def merge_labeled_cues(
    cues: list[Cue],
    kinds: dict[int, str],
    *,
    speech_gap_seconds: float,
    max_speech_chars: int = DEFAULT_MAX_SPEECH_CHARS,
    max_speech_seconds: float = DEFAULT_MAX_SPEECH_SECONDS,
) -> list[Segment]:
    if not cues:
        return []
    groups: list[tuple[str, list[Cue]]] = []
    current_kind = kinds.get(cues[0].cue_id, "speech")
    current = [cues[0]]
    for cue in cues[1:]:
        kind = kinds.get(cue.cue_id, "speech")
        join_song = current_kind == "song" and kind == "song"
        join_speech = current_kind == "speech" and kind == "speech" and _can_join_speech(
            current,
            cue,
            gap_seconds=speech_gap_seconds,
            max_chars=max_speech_chars,
            max_seconds=max_speech_seconds,
        )
        if join_song or join_speech:
            current.append(cue)
            continue
        groups.append((current_kind, current))
        current_kind = kind
        current = [cue]
    groups.append((current_kind, current))
    return [_to_segment(index + 1, group, kind=kind) for index, (kind, group) in enumerate(groups)]


def _can_join_speech(
    current: list[Cue],
    cue: Cue,
    *,
    gap_seconds: float,
    max_chars: int,
    max_seconds: float,
) -> bool:
    previous = current[-1]
    if cue.start - previous.end > gap_seconds:
        return False
    joined_chars = sum(len(item.text) for item in current) + 1 + len(cue.text)
    if joined_chars > max_chars:
        return False
    if cue.end - current[0].start > max_seconds:
        return False
    return True


def _to_segment(segment_id: int, cues: list[Cue], *, kind: str = "speech") -> Segment:
    text = " ".join(cue.text for cue in cues)
    return Segment(
        segment_id=segment_id,
        start=cues[0].start,
        end=cues[-1].end,
        text=text,
        cue_ids=tuple(cue.cue_id for cue in cues),
        kind=kind,
    )
