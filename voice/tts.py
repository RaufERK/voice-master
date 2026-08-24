from __future__ import annotations

from pathlib import Path

from .checkpoint import tts_path


TTS_INSTRUCTIONS = (
    "Speak in clear, calm Russian. Solemn spiritual lecture, unhurried, "
    "not theatrical, not a commercial voice."
)


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
    return dest


def concatenate_mp3(paths: list[Path], output_path: Path) -> None:
    import subprocess
    import tempfile

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


def segment_audio_path(work_dir: Path, segment_id: int) -> Path:
    return tts_path(work_dir, segment_id)
