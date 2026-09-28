from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from openai import OpenAI

from . import DEFAULT_ASR_MODEL, PROJECT_ROOT
from .audio import probe_duration
from .checkpoint import load_json, save_json


AUDIO_SUFFIXES = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".aac"}

# Whisper hears this Summit fiat as "bind the willfulness".
_ASR_PHRASE_FIXES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"bind the willfulness", re.IGNORECASE), "bolts of blue lightning"),
)


class TranscribeError(RuntimeError):
    pass


def list_audio_files(folder: Path) -> list[Path]:
    files = [
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES and not path.name.startswith(".")
    ]
    return sorted(files, key=lambda path: path.name.casefold())


def transcribe_folder(
    folder: Path,
    *,
    api_key: str,
    base_url: str,
    model: str = DEFAULT_ASR_MODEL,
    force: bool = False,
    output_path: Path | None = None,
) -> Path:
    if not folder.is_dir():
        raise TranscribeError(f"Not a directory: {folder}")
    files = list_audio_files(folder)
    if not files:
        raise TranscribeError(f"No audio files in {folder}")

    work_dir = PROJECT_ROOT / "work" / folder.name / "asr"
    work_dir.mkdir(parents=True, exist_ok=True)
    dest = output_path or (PROJECT_ROOT / "output" / f"{folder.name}_en.md")
    dest.parent.mkdir(parents=True, exist_ok=True)

    client = OpenAI(
        api_key=api_key,
        base_url=base_url.rstrip("/"),
        timeout=600.0,
        max_retries=2,
    )
    records: list[dict[str, Any]] = []
    for index, path in enumerate(files, start=1):
        cache_path = work_dir / f"{_safe_stem(path)}.json"
        print(f"         asr {index}/{len(files)} {path.name}")
        record = _transcribe_file(client, path, cache_path=cache_path, model=model, force=force)
        records.append(record)

    dest.write_text(_render_markdown(folder.name, records), encoding="utf-8")
    return dest


def _transcribe_file(
    client: OpenAI,
    path: Path,
    *,
    cache_path: Path,
    model: str,
    force: bool,
) -> dict[str, Any]:
    existing = load_json(cache_path)
    if existing and existing.get("ok") and not force:
        print("         cached")
        return _fix_cached_record(cache_path, existing)

    duration = probe_duration(path)
    with path.open("rb") as handle:
        try:
            result = client.audio.transcriptions.create(
                model=model,
                file=(path.name, handle),
                language="en",
                response_format="verbose_json",
            )
        except Exception as exc:
            raise TranscribeError(f"ASR failed for {path.name}: {exc}") from exc

    text = apply_asr_fixes(str(getattr(result, "text", "") or "").strip())
    segments = _segments_from_result(result)
    if not text and segments:
        text = " ".join(item["text"] for item in segments).strip()
    if not text:
        raise TranscribeError(f"Empty transcript for {path.name}")

    record = {
        "ok": True,
        "file": path.name,
        "duration": duration,
        "model": model,
        "text": text,
        "segments": segments,
    }
    save_json(cache_path, record)
    return record


def apply_asr_fixes(text: str) -> str:
    for pattern, right in _ASR_PHRASE_FIXES:
        text = pattern.sub(lambda match, replacement=right: _match_case(match.group(0), replacement), text)
    return text


def _match_case(found: str, replacement: str) -> str:
    if found[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def _fix_cached_record(cache_path: Path, record: dict[str, Any]) -> dict[str, Any]:
    text = apply_asr_fixes(str(record.get("text") or ""))
    segments = []
    for item in record.get("segments") or []:
        segment = dict(item)
        segment["text"] = apply_asr_fixes(str(segment.get("text") or ""))
        segments.append(segment)
    if text == record.get("text") and segments == (record.get("segments") or []):
        return record
    fixed = {**record, "text": text, "segments": segments}
    save_json(cache_path, fixed)
    return fixed


def _segments_from_result(result: object) -> list[dict[str, Any]]:
    raw = getattr(result, "segments", None) or []
    segments: list[dict[str, Any]] = []
    for item in raw:
        text = apply_asr_fixes(str(getattr(item, "text", "") or "").strip())
        if not text:
            continue
        segments.append(
            {
                "start": float(getattr(item, "start", 0.0) or 0.0),
                "end": float(getattr(item, "end", 0.0) or 0.0),
                "text": text,
            }
        )
    return segments


def _render_markdown(title: str, records: list[dict[str, Any]]) -> str:
    total = sum(float(item.get("duration") or 0.0) for item in records)
    lines = [
        f"# {title} — English transcripts",
        "",
        f"{len(records)} clips · {_format_clock(total)} total. "
        "Each heading is one source file; timestamps are inside the clip.",
        "",
        "## Contents",
        "",
    ]
    for record in records:
        name = str(record["file"])
        duration = _format_clock(float(record.get("duration") or 0.0))
        lines.append(f"- [{name}](#{_clip_id(name)}) — {duration}")
    lines.extend(["", "---", ""])

    for index, record in enumerate(records):
        name = str(record["file"])
        duration = _format_clock(float(record.get("duration") or 0.0))
        lines.append(f'<a id="{_clip_id(name)}"></a>')
        lines.append("")
        lines.append(f"## {name}")
        lines.append("")
        lines.append(f"**File:** `{name}`  ")
        lines.append(f"**Duration:** {duration}")
        lines.append("")
        segments = record.get("segments") or []
        if segments:
            for segment in segments:
                stamp = _format_clock(float(segment.get("start") or 0.0))
                lines.append(f"**[{stamp}]** {segment['text']}")
                lines.append("")
        else:
            lines.append(str(record.get("text") or "").strip())
            lines.append("")
        if index < len(records) - 1:
            lines.extend(["---", ""])
    return "\n".join(lines).rstrip() + "\n"


def _safe_stem(path: Path) -> str:
    return "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in path.stem)


def _clip_id(name: str) -> str:
    prefix = name.split(" ", 1)[0]
    if prefix.isdigit():
        return f"clip-{prefix}"
    return "clip-" + "".join(ch.lower() if ch.isalnum() else "-" for ch in name).strip("-")


def _format_clock(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"
