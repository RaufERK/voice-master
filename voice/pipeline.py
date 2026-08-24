from __future__ import annotations

from pathlib import Path

from . import (
    DEFAULT_CONTEXT_SEGMENTS,
    DEFAULT_GAP_MERGE_SECONDS,
    DEFAULT_MAX_RETRIES,
    DEFAULT_TTS_FALLBACK_MODEL,
    AppConfig,
    Glossary,
    PROJECT_ROOT,
    Segment,
)
from .checkpoint import batch_path, load_done_translations, load_json, save_json, tts_path
from .sbv import load_sbv
from .segments import filter_window, merge_cues
from .translate import OpenAITranslator, TranslationError
from .tts import concatenate_mp3, synthesize_segment


def work_dir_for(start: float, duration: float) -> Path:
    start_tag = int(start)
    end_tag = int(start + duration)
    return PROJECT_ROOT / "work" / f"window_{start_tag}_{end_tag}"


def output_path_for(start: float, duration: float) -> Path:
    start_tag = int(start)
    end_tag = int(start + duration)
    return PROJECT_ROOT / "output" / f"lecture_ru_{start_tag}_{end_tag}s.mp3"


def run_pipeline(config: AppConfig, glossary: Glossary | None) -> str:
    end = config.start_seconds + config.duration_seconds
    work_dir = work_dir_for(config.start_seconds, config.duration_seconds)
    work_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] Parsing captions...")
    cues = load_sbv(config.captions_path)
    window = filter_window(cues, config.start_seconds, end)
    segments = merge_cues(window, gap_seconds=DEFAULT_GAP_MERGE_SECONDS)
    if not segments:
        raise RuntimeError("No caption segments in this time window")
    print(f"         {len(window)} cues → {len(segments)} segments")
    save_json(
        work_dir / "meta.json",
        {
            "captions": str(config.captions_path),
            "start": config.start_seconds,
            "end": end,
            "model": config.model,
            "tts_model": config.tts_model,
            "segment_count": len(segments),
        },
    )

    translator = OpenAITranslator(
        api_key=config.openai_api_key,
        base_url=config.openai_base_url,
        model=config.model,
    )

    print("[2/5] Translating...")
    translations = translate_segments(work_dir, segments, config, glossary, translator)

    ru_text_path = work_dir / "ru_segments.json"
    save_json(
        ru_text_path,
        {
            "segments": [
                {
                    "id": segment.segment_id,
                    "start": segment.start,
                    "end": segment.end,
                    "en": segment.text,
                    "ru": translations[segment.segment_id],
                }
                for segment in segments
            ]
        },
    )
    print(f"         wrote {ru_text_path.relative_to(PROJECT_ROOT)}")

    if config.translate_only:
        print("[3/5] Translate-only: skipping TTS")
        return f"translated: {len(segments)} segments"

    print("[3/5] Synthesizing speech...")
    audio_paths = synthesize_segments(translator.client, work_dir, segments, translations, config)

    print("[4/5] Concatenating...")
    output_path = output_path_for(config.start_seconds, config.duration_seconds)
    concatenate_mp3(audio_paths, output_path)
    print(f"         wrote {output_path.relative_to(PROJECT_ROOT)}")

    print("[5/5] Done")
    return f"ok: {output_path.name}"


def translate_segments(
    work_dir: Path,
    segments: list[Segment],
    config: AppConfig,
    glossary: Glossary | None,
    translator: OpenAITranslator,
) -> dict[int, str]:
    translations = load_done_translations(work_dir) if not config.force else {}
    batches = [
        segments[index : index + config.batch_size]
        for index in range(0, len(segments), config.batch_size)
    ]
    for batch_index, batch in enumerate(batches):
        batch_ids = [segment.segment_id for segment in batch]
        existing_path = batch_path(work_dir, batch_index)
        existing = load_json(existing_path)
        if existing and existing.get("ok") and not config.force:
            print(f"         batch {batch_index + 1}/{len(batches)} cached")
            continue

        items = [{"id": segment.segment_id, "text": segment.text} for segment in batch]
        previous = _previous_context(segments, translations, batch[0].segment_id)
        print(
            f"         batch {batch_index + 1}/{len(batches)} "
            f"(segments {batch_ids[0]}-{batch_ids[-1]})"
        )
        translated: list[dict[str, str | int]] | None = None
        last_error: Exception | None = None
        for attempt in range(1, DEFAULT_MAX_RETRIES + 1):
            try:
                raw = translator.translate_batch(items, previous=previous, glossary=glossary)
                translated = _validate_batch(batch_ids, raw)
                break
            except Exception as exc:
                last_error = exc
                print(f"         retry {attempt}/{DEFAULT_MAX_RETRIES}: {exc}")

        if translated is None:
            save_json(existing_path, {"ok": False, "ids": batch_ids, "error": str(last_error)})
            raise TranslationError(
                f"Batch {batch_index + 1} failed after {DEFAULT_MAX_RETRIES} retries: {last_error}"
            )

        save_json(
            existing_path,
            {"ok": True, "ids": batch_ids, "source": items, "translated": translated},
        )
        for item in translated:
            translations[int(item["id"])] = str(item["text"])

    missing = [segment.segment_id for segment in segments if segment.segment_id not in translations]
    if missing:
        raise TranslationError(f"Missing translations for segments: {missing[:12]}")
    return translations


def synthesize_segments(
    client,
    work_dir: Path,
    segments: list[Segment],
    translations: dict[int, str],
    config: AppConfig,
) -> list[Path]:
    paths: list[Path] = []
    for index, segment in enumerate(segments, start=1):
        dest = tts_path(work_dir, segment.segment_id)
        print(f"         tts {index}/{len(segments)} segment {segment.segment_id}")
        synthesize_segment(
            client,
            text=translations[segment.segment_id],
            model=config.tts_model,
            voice=config.tts_voice,
            fallback_model=DEFAULT_TTS_FALLBACK_MODEL,
            dest=dest,
            force=config.force,
        )
        paths.append(dest)
    return paths


def _previous_context(
    segments: list[Segment],
    translations: dict[int, str],
    first_id: int,
) -> list[dict[str, str | int]]:
    previous: list[dict[str, str | int]] = []
    for segment in segments:
        if segment.segment_id >= first_id:
            break
        ru = translations.get(segment.segment_id)
        if ru is None:
            continue
        previous.append({"id": segment.segment_id, "en": segment.text, "ru": ru})
    return previous[-DEFAULT_CONTEXT_SEGMENTS:]


def _validate_batch(expected_ids: list[int], raw: list[dict[str, object]]) -> list[dict[str, str | int]]:
    got: dict[int, str] = {}
    for item in raw:
        item_id = int(item["id"])
        text = str(item.get("text") or "").strip()
        if not text:
            raise TranslationError(f"Empty translation for id {item_id}")
        got[item_id] = text
    missing = [item_id for item_id in expected_ids if item_id not in got]
    extra = [item_id for item_id in got if item_id not in expected_ids]
    if missing or extra:
        raise TranslationError(f"ID mismatch: missing={missing[:12]} extra={extra[:12]}")
    return [{"id": item_id, "text": got[item_id]} for item_id in expected_ids]
