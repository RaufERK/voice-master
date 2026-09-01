from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from .checkpoint import tts_path


TTS_INSTRUCTIONS = (
    "Speak in clear, calm Russian. Solemn spiritual lecture, unhurried, "
    "not theatrical, not a commercial voice."
)
TTS_MAX_CHARS = 3500


class TtsError(RuntimeError):
    pass


def synthesize_segment(
    client,
    *,
    text: str,
    model: str,
    voice: str,
    fallback_model: str,
    dest: Path,
    force: bool,
) -> Path:
    if dest.exists() and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    chunks = _split_for_tts(text)
    if len(chunks) == 1:
        _synthesize_one(
            client,
            text=chunks[0],
            model=model,
            voice=voice,
            fallback_model=fallback_model,
            dest=dest,
        )
        return dest

    part_paths: list[Path] = []
    for index, chunk in enumerate(chunks, start=1):
        part = dest.with_name(f"{dest.stem}_p{index:02d}.mp3")
        if force or not part.exists():
            _synthesize_one(
                client,
                text=chunk,
                model=model,
                voice=voice,
                fallback_model=fallback_model,
                dest=part,
            )
        part_paths.append(part)
    concatenate_mp3(part_paths, dest)
    return dest


def _synthesize_one(
    client,
    *,
    text: str,
    model: str,
    voice: str,
    fallback_model: str,
    dest: Path,
) -> None:
    try:
        _write_speech(client, text=text, model=model, voice=voice, dest=dest)
    except Exception as exc:
        if model == fallback_model:
            raise TtsError(f"TTS failed ({model}): {exc}") from exc
        print(f"         TTS {model} failed, trying {fallback_model}: {exc}")
        try:
            _write_speech(client, text=text, model=fallback_model, voice=voice, dest=dest)
        except Exception as fallback_exc:
            raise TtsError(f"TTS failed ({fallback_model}): {fallback_exc}") from fallback_exc


def concatenate_mp3(paths: list[Path], output_path: Path) -> None:
    if not paths:
        raise TtsError("No TTS clips to concatenate")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as handle:
        list_path = Path(handle.name)
        for path in paths:
            escaped = str(path.resolve()).replace("'", r"'\''")
            handle.write(f"file '{escaped}'\n")
    try:
        subprocess.run(
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
                "libmp3lame",
                "-q:a",
                "3",
                str(output_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise TtsError(f"ffmpeg concat failed: {exc.stderr[-800:]}") from exc
    finally:
        list_path.unlink(missing_ok=True)


def _write_speech(client, *, text: str, model: str, voice: str, dest: Path) -> None:
    kwargs: dict[str, str] = {
        "model": model,
        "voice": voice,
        "input": text,
        "response_format": "mp3",
    }
    if _supports_instructions(model):
        kwargs["instructions"] = TTS_INSTRUCTIONS
    response = client.audio.speech.create(**kwargs)
    dest.write_bytes(response.content)


def _supports_instructions(model: str) -> bool:
    lowered = model.lower()
    return "mini-tts" in lowered or lowered.startswith("gpt-4o")


def _split_for_tts(text: str, max_chars: int = TTS_MAX_CHARS) -> list[str]:
    stripped = text.strip()
    if len(stripped) <= max_chars:
        return [stripped]
    chunks: list[str] = []
    remaining = stripped
    while remaining:
        if len(remaining) <= max_chars:
            chunks.append(remaining)
            break
        window = remaining[:max_chars]
        cut = max(window.rfind(". "), window.rfind("! "), window.rfind("? "), window.rfind("… "))
        if cut < max_chars * 0.4:
            cut = window.rfind(" ")
        if cut < 1:
            cut = max_chars
            piece, remaining = remaining[:cut], remaining[cut:]
        else:
            piece, remaining = remaining[: cut + 1], remaining[cut + 1 :]
        piece = piece.strip()
        remaining = remaining.strip()
        if piece:
            chunks.append(piece)
    return chunks or [stripped]


def segment_audio_path(work_dir: Path, segment_id: int) -> Path:
    return tts_path(work_dir, segment_id)
