from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE_URL = "https://spoken-word.info/openai-proxy/v1"
DEFAULT_MODEL = "gpt-5.6-luna"
DEFAULT_TTS_MODEL = "tts-1-hd"
DEFAULT_TTS_VOICE = "nova"
DEFAULT_TTS_FALLBACK_MODEL = "tts-1-hd"
DEFAULT_BATCH_SIZE = 12
DEFAULT_CONTEXT_SEGMENTS = 4
DEFAULT_MAX_RETRIES = 3
DEFAULT_DURATION_SECONDS = 300.0
DEFAULT_GAP_MERGE_SECONDS = 1.2
DEFAULT_GLOSSARY = PROJECT_ROOT / "glossaries" / "lecture.yaml"
DEFAULT_CAPTIONS = PROJECT_ROOT / "audio-source" / "LECTURE_1" / "captions.sbv"


@dataclass(frozen=True)
class Cue:
    cue_id: int
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Segment:
    segment_id: int
    start: float
    end: float
    text: str
    cue_ids: tuple[int, ...]


@dataclass(frozen=True)
class Glossary:
    series: str
    context: str
    names: dict[str, str]
    terms: dict[str, str]
    notes: list[str]


@dataclass(frozen=True)
class AppConfig:
    openai_api_key: str
    openai_base_url: str
    model: str
    tts_model: str
    tts_voice: str
    glossary_path: Path | None
    captions_path: Path
    start_seconds: float
    duration_seconds: float
    batch_size: int
    force: bool
    translate_only: bool
