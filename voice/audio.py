from __future__ import annotations

import subprocess
from pathlib import Path

from . import DEFAULT_SAMPLE_RATE


class AudioError(RuntimeError):
    pass


def find_source_media(captions_path: Path) -> Path:
    folder = captions_path.parent
    matches = sorted(folder.glob("*.mp4")) + sorted(folder.glob("*.mkv")) + sorted(folder.glob("*.wav"))
    if not matches:
        raise AudioError(f"No source media next to captions: {folder}")
    return matches[0]


def probe_duration(path: Path) -> float:
    result = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            str(path),
        ]
    )
    return float(result.stdout.strip())


def extract_window_wav(source: Path, dest: Path, *, start: float, duration: float) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{duration:.3f}",
            "-i",
            str(source),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(DEFAULT_SAMPLE_RATE),
            str(dest),
        ]
    )
    return dest


def slice_wav(source: Path, dest: Path, *, start: float, duration: float) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if duration <= 0.02:
        raise AudioError(f"Refusing empty slice: {duration}")
    _run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{duration:.3f}",
            "-i",
            str(source),
            "-ac",
            "1",
            "-ar",
            str(DEFAULT_SAMPLE_RATE),
            str(dest),
        ]
    )
    return dest


def to_wav(source: Path, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(source),
            "-ac",
            "1",
            "-ar",
            str(DEFAULT_SAMPLE_RATE),
            str(dest),
        ]
    )
    return dest


def duck_wav(source: Path, dest: Path, *, duck_db: float) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(source),
            "-af",
            f"volume={duck_db}dB",
            str(dest),
        ]
    )
    return dest


def mix_wav(foreground: Path, background: Path, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(foreground),
            "-i",
            str(background),
            "-filter_complex",
            "amix=inputs=2:duration=first:dropout_transition=0:normalize=0",
            "-ac",
            "1",
            "-ar",
            str(DEFAULT_SAMPLE_RATE),
            str(dest),
        ]
    )
    return dest


def trim_wav(source: Path, dest: Path, *, duration: float) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(source),
            "-t",
            f"{duration:.3f}",
            "-ac",
            "1",
            "-ar",
            str(DEFAULT_SAMPLE_RATE),
            str(dest),
        ]
    )
    return dest


def skip_wav(source: Path, dest: Path, *, start: float) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-i",
            str(source),
            "-ac",
            "1",
            "-ar",
            str(DEFAULT_SAMPLE_RATE),
            str(dest),
        ]
    )
    return dest


def concat_wavs(paths: list[Path], dest: Path) -> Path:
    import tempfile

    if not paths:
        raise AudioError("Nothing to concatenate")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as handle:
        list_path = Path(handle.name)
        for path in paths:
            escaped = str(path.resolve()).replace("'", r"'\''")
            handle.write(f"file '{escaped}'\n")
    try:
        _run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_path),
                "-c:a",
                "pcm_s16le",
                "-ar",
                str(DEFAULT_SAMPLE_RATE),
                "-ac",
                "1",
                str(dest),
            ]
        )
    finally:
        list_path.unlink(missing_ok=True)
    return dest


def wav_to_mp3(source: Path, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(source),
            "-c:a",
            "libmp3lame",
            "-q:a",
            "3",
            str(dest),
        ]
    )
    return dest


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        tail = (result.stderr or result.stdout)[-800:]
        raise AudioError(f"{command[0]} failed: {tail}")
    return result
