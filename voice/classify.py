from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from . import DEFAULT_CLASSIFY_BATCH_SIZE, Cue
from .checkpoint import load_json, save_json
from .translate import OpenAITranslator, TranslationError


KindName = Literal["speech", "song"]


class ClassifiedItem(BaseModel):
    id: int
    kind: KindName


class ClassifyBatch(BaseModel):
    items: list[ClassifiedItem] = Field(min_length=1)


CLASSIFY_PROMPT = """You label YouTube caption cues from a spiritual lecture.

kind=speech: spoken lecture, prayer, invocation, announcements (please be seated, song number…).
kind=song: hymn or sung verse / lyrics, even if ASR is broken.

Keep each id. Do not drop or add ids. Return only structured labels.
"""


def classify_cues(
    translator: OpenAITranslator,
    cues: list[Cue],
    *,
    work_dir: Path,
    force: bool,
    batch_size: int = DEFAULT_CLASSIFY_BATCH_SIZE,
) -> dict[int, str]:
    cache_path = work_dir / "kinds.json"
    cached = load_json(cache_path)
    if cached and cached.get("ok") and not force:
        raw = {int(key): str(value) for key, value in (cached.get("kinds") or {}).items()}
        if all(cue.cue_id in raw for cue in cues):
            print("         classification cached")
            return raw

    kinds: dict[int, str] = {}
    batches = [cues[index : index + batch_size] for index in range(0, len(cues), batch_size)]
    kinds_dir = work_dir / "kinds_batches"
    kinds_dir.mkdir(parents=True, exist_ok=True)

    for batch_index, batch in enumerate(batches):
        batch_cache = kinds_dir / f"{batch_index:04d}.json"
        existing = load_json(batch_cache)
        if existing and existing.get("ok") and not force:
            for key, value in (existing.get("kinds") or {}).items():
                kinds[int(key)] = str(value)
            print(f"         classify batch {batch_index + 1}/{len(batches)} cached")
            continue

        print(f"         classify batch {batch_index + 1}/{len(batches)}")
        context = _context_items(cues, kinds, batch[0].cue_id)
        try:
            parsed = _classify_batch(translator, batch, context=context)
            expected = [cue.cue_id for cue in batch]
            batch_kinds = _validate_kinds(expected, parsed)
        except Exception as exc:
            print(f"         classify batch {batch_index + 1} API failed ({exc}); using local heuristic")
            batch_kinds = _heuristic_kinds(batch, start_mode=_last_kind(kinds) or "speech")

        save_json(batch_cache, {"ok": True, "kinds": {str(key): value for key, value in batch_kinds.items()}})
        kinds.update(batch_kinds)

    save_json(cache_path, {"ok": True, "kinds": {str(key): value for key, value in kinds.items()}})
    song_count = sum(1 for value in kinds.values() if value == "song")
    print(f"         {len(kinds) - song_count} speech / {song_count} song cues")
    return kinds


def _classify_batch(
    translator: OpenAITranslator,
    cues: list[Cue],
    *,
    context: list[dict[str, str | int]],
) -> list[dict[str, object]]:
    items = [{"id": cue.cue_id, "text": cue.text} for cue in cues]
    payload = {"previous_labels": context, "cues": items}
    messages = [
        {"role": "system", "content": CLASSIFY_PROMPT},
        {
            "role": "user",
            "content": (
                "Label each cue in 'cues' as speech or song. "
                "previous_labels is context only — do not relabel those ids. "
                "Return JSON {items: [{id, kind}]}.\n\n"
                + _to_json(payload)
            ),
        },
    ]
    return _parse_kinds(translator, messages)


def _parse_kinds(translator: OpenAITranslator, messages: list[dict[str, str]]) -> list[dict[str, object]]:
    try:
        completion = translator.client.chat.completions.parse(
            model=translator.model,
            messages=messages,
            response_format=ClassifyBatch,
        )
        message = completion.choices[0].message
        parsed = message.parsed
        if parsed is None:
            raise TranslationError("Empty classification")
        return [{"id": item.id, "kind": item.kind} for item in parsed.items]
    except Exception:
        fallback = _parse_kinds_fallback(translator, messages)
        if fallback is None:
            raise
        return fallback


def _parse_kinds_fallback(
    translator: OpenAITranslator,
    messages: list[dict[str, str]],
) -> list[dict[str, object]] | None:
    try:
        completion = translator.client.chat.completions.create(
            model=translator.model,
            messages=messages,
            response_format={"type": "json_object"},
        )
        content = completion.choices[0].message.content or ""
        match = re.search(r"\{[\s\S]*\}", content)
        payload = json.loads(match.group(0) if match else content)
        parsed = ClassifyBatch.model_validate(payload)
        return [{"id": item.id, "kind": item.kind} for item in parsed.items]
    except Exception:
        return None


def _validate_kinds(expected_ids: list[int], raw: list[dict[str, object]]) -> dict[int, str]:
    got: dict[int, str] = {}
    for item in raw:
        cue_id = int(item["id"])
        kind = str(item.get("kind") or "").strip()
        if kind not in {"speech", "song"}:
            raise TranslationError(f"Bad kind for id {cue_id}: {kind}")
        got[cue_id] = kind
    missing = [cue_id for cue_id in expected_ids if cue_id not in got]
    if missing:
        raise TranslationError(f"Missing kinds for cue ids: {missing[:12]}")
    return {cue_id: got[cue_id] for cue_id in expected_ids}


def _heuristic_kinds(cues: list[Cue], *, start_mode: str = "speech") -> dict[int, str]:
    kinds: dict[int, str] = {}
    mode = start_mode if start_mode in {"speech", "song"} else "speech"
    for cue in cues:
        lower = cue.text.lower()
        if any(
            marker in lower
            for marker in (
                "won't you be seated",
                "karmic board",
                "pearls of wisdom",
                "i am that i am",
            )
        ):
            mode = "speech"
        if any(
            marker in lower
            for marker in (
                "god of mercy, love",
                "we love thee and thy name",
                "mercy is the grace of love",
            )
        ):
            mode = "song"
        if "song number" in lower:
            kinds[cue.cue_id] = "speech"
            mode = "song"
            continue
        kinds[cue.cue_id] = mode
        if "holy spirit" in lower and "amen" in lower:
            mode = "song"
    return kinds


def _context_items(
    cues: list[Cue],
    kinds: dict[int, str],
    first_id: int,
    *,
    limit: int = 4,
) -> list[dict[str, str | int]]:
    previous: list[dict[str, str | int]] = []
    for cue in cues:
        if cue.cue_id >= first_id:
            break
        kind = kinds.get(cue.cue_id)
        if kind is None:
            continue
        previous.append({"id": cue.cue_id, "text": cue.text, "kind": kind})
    return previous[-limit:]


def _last_kind(kinds: dict[int, str]) -> str | None:
    if not kinds:
        return None
    return kinds[max(kinds)]


def _to_json(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)
