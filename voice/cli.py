from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from . import (
    DEFAULT_ASR_MODEL,
    DEFAULT_BASE_URL,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CAPTIONS,
    DEFAULT_DUCK_DB,
    DEFAULT_DURATION_SECONDS,
    DEFAULT_GLOSSARY,
    DEFAULT_MODEL,
    DEFAULT_TTS_MODEL,
    DEFAULT_TTS_VOICE,
    PROJECT_ROOT,
    AppConfig,
)
from .audio import AudioError, find_source_media, probe_duration
from .glossary import load_glossary
from .pipeline import run_pipeline
from .sbv import load_sbv
from .transcribe import TranscribeError, transcribe_folder
from .translate import TranslationError
from .tts import TtsError


def main(argv: list[str] | None = None) -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    args = parse_args(argv)

    if args.check:
        return run_check()
    if args.transcribe:
        return run_transcribe(args)

    captions_path = Path(args.captions).expanduser()
    if not captions_path.is_absolute():
        captions_path = PROJECT_ROOT / captions_path
    if not captions_path.exists():
        print(f"Captions not found: {captions_path}", file=sys.stderr)
        return 1

    source_path = None
    if args.source:
        source_path = Path(args.source).expanduser()
        if not source_path.is_absolute():
            source_path = PROJECT_ROOT / source_path
    elif args.en_bed or args.duration <= 0:
        try:
            source_path = find_source_media(captions_path)
        except AudioError as exc:
            if args.en_bed:
                print(f"ERROR: {exc}", file=sys.stderr)
                return 1

    duration_seconds = args.duration
    if duration_seconds <= 0:
        duration_seconds = _full_duration(captions_path, source_path, args.start)
        if duration_seconds <= 0:
            print("ERROR: empty time window", file=sys.stderr)
            return 1

    config = AppConfig(
        openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
        openai_base_url=(os.getenv("OPENAI_BASE_URL") or DEFAULT_BASE_URL).strip(),
        model=(args.model or os.getenv("TRANSLATION_MODEL") or DEFAULT_MODEL).strip(),
        tts_model=(args.tts_model or os.getenv("TTS_MODEL") or DEFAULT_TTS_MODEL).strip(),
        tts_voice=(args.voice or os.getenv("TTS_VOICE") or DEFAULT_TTS_VOICE).strip(),
        glossary_path=_resolve_glossary(args.glossary, args.no_glossary),
        captions_path=captions_path,
        source_path=source_path,
        start_seconds=args.start,
        duration_seconds=duration_seconds,
        batch_size=args.batch_size,
        force=args.force,
        translate_only=args.translate_only,
        en_bed=args.en_bed,
        duck_db=args.duck_db,
    )
    if not config.openai_api_key:
        print("OPENAI_API_KEY is missing in .env", file=sys.stderr)
        return 1

    glossary = load_glossary(config.glossary_path)
    if config.glossary_path:
        print(f"Glossary: {config.glossary_path}")
    print(
        f"Window: {config.start_seconds:.0f}s + {config.duration_seconds:.0f}s  "
        f"model={config.model}  tts={config.tts_model}/{config.tts_voice}  "
        f"en_bed={config.en_bed}"
    )
    try:
        print(run_pipeline(config, glossary))
    except (TranslationError, TtsError, AudioError, RuntimeError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Translate lecture captions and synthesize Russian audio")
    parser.add_argument("--check", action="store_true", help="Check OpenAI proxy credentials and exit")
    parser.add_argument(
        "--transcribe",
        metavar="DIR",
        help="Transcribe audio files in DIR into one English markdown file",
    )
    parser.add_argument("--captions", default=str(DEFAULT_CAPTIONS), help="SBV captions path")
    parser.add_argument("--start", type=float, default=0.0, help="Window start in seconds")
    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_DURATION_SECONDS,
        help="Window length in seconds (default 300; 0 = full lecture)",
    )
    parser.add_argument("--model", help="Translation model")
    parser.add_argument("--tts-model", help="TTS model")
    parser.add_argument("--voice", help="TTS voice")
    parser.add_argument("--glossary", help="YAML glossary path")
    parser.add_argument("--no-glossary", action="store_true", help="Translate without a glossary")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--translate-only", action="store_true", help="Stop after translation")
    parser.add_argument(
        "--en-bed",
        action="store_true",
        help="Quiet original English under Russian; pause EN when Russian is longer; songs passthrough",
    )
    parser.add_argument("--source", help="Source lecture media (default: mp4 next to captions)")
    parser.add_argument("--duck-db", type=float, default=DEFAULT_DUCK_DB, help="English bed gain in dB")
    parser.add_argument("--force", action="store_true", help="Redo cached translation and TTS")
    return parser.parse_args(argv)


def run_transcribe(args: argparse.Namespace) -> int:
    folder = Path(args.transcribe).expanduser()
    if not folder.is_absolute():
        folder = PROJECT_ROOT / folder
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        print("OPENAI_API_KEY is missing in .env", file=sys.stderr)
        return 1
    base_url = (os.getenv("OPENAI_BASE_URL") or DEFAULT_BASE_URL).strip()
    model = (args.model or os.getenv("ASR_MODEL") or DEFAULT_ASR_MODEL).strip()
    print(f"Transcribe: {folder}  model={model}")
    try:
        dest = transcribe_folder(
            folder,
            api_key=api_key,
            base_url=base_url,
            model=model,
            force=args.force,
        )
    except (TranscribeError, AudioError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"ok: {dest.relative_to(PROJECT_ROOT)}")
    return 0


def run_check() -> int:
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    base_url = (os.getenv("OPENAI_BASE_URL") or DEFAULT_BASE_URL).strip()
    if not api_key:
        print("OPENAI_API_KEY is missing", file=sys.stderr)
        return 1
    client = OpenAI(api_key=api_key, base_url=base_url.rstrip("/"), timeout=30.0)
    try:
        models = client.models.list()
    except Exception as exc:
        print(f"Proxy check failed: {exc}", file=sys.stderr)
        return 1
    ids = [item.id for item in models.data][:16]
    print(f"Proxy OK: {base_url}")
    print("Models:", ", ".join(ids) if ids else "(none listed)")
    return 0


def _resolve_glossary(value: str | None, no_glossary: bool) -> Path | None:
    if no_glossary:
        return None
    if value:
        path = Path(value).expanduser()
        return path if path.is_absolute() else PROJECT_ROOT / path
    return DEFAULT_GLOSSARY if DEFAULT_GLOSSARY.exists() else None


def _full_duration(captions_path: Path, source_path: Path | None, start_seconds: float) -> float:
    if source_path is not None:
        return probe_duration(source_path) - start_seconds
    cues = load_sbv(captions_path)
    if not cues:
        return 0.0
    return max(cue.end for cue in cues) - start_seconds
