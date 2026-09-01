from __future__ import annotations

from pathlib import Path

from . import DEFAULT_DUCK_DB, Segment
from .audio import (
    AudioError,
    concat_wavs,
    duck_wav,
    mix_wav,
    probe_duration,
    skip_wav,
    slice_wav,
    to_wav,
    trim_wav,
    wav_to_mp3,
)
from .checkpoint import tts_path


def mix_en_bed(
    *,
    source_wav: Path,
    segments: list[Segment],
    window_start: float,
    window_end: float,
    work_dir: Path,
    output_mp3: Path,
    duck_db: float = DEFAULT_DUCK_DB,
) -> Path:
    parts_dir = work_dir / "mix_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    pieces: list[Path] = []
    cursor = window_start
    part_index = 0

    def add_original(start: float, end: float) -> None:
        nonlocal part_index
        duration = end - start
        if duration <= 0.04:
            return
        dest = parts_dir / f"{part_index:04d}_orig.wav"
        # source_wav begins at window_start
        slice_wav(source_wav, dest, start=start - window_start, duration=duration)
        pieces.append(dest)
        part_index += 1

    for segment in segments:
        if segment.start > cursor + 0.04:
            add_original(cursor, segment.start)
        if segment.kind == "song":
            add_original(segment.start, min(segment.end, window_end))
        elif segment.kind == "speech":
            dest = parts_dir / f"{part_index:04d}_speech.wav"
            _mix_speech(
                source_wav=source_wav,
                window_start=window_start,
                window_end=window_end,
                segment=segment,
                work_dir=work_dir,
                dest=dest,
                duck_db=duck_db,
            )
            pieces.append(dest)
            part_index += 1
        else:
            add_original(segment.start, segment.end)
        cursor = max(cursor, segment.end)

    if window_end > cursor + 0.04:
        add_original(cursor, window_end)
    if not pieces:
        raise AudioError("Mix produced no audio pieces")

    mixed = work_dir / "mixed.wav"
    concat_wavs(pieces, mixed)
    wav_to_mp3(mixed, output_mp3)
    return output_mp3


def _mix_speech(
    *,
    source_wav: Path,
    window_start: float,
    window_end: float,
    segment: Segment,
    work_dir: Path,
    dest: Path,
    duck_db: float,
) -> None:
    tts_mp3 = tts_path(work_dir, segment.segment_id)
    if not tts_mp3.exists():
        raise AudioError(f"Missing TTS for segment {segment.segment_id}")
    ru_wav = dest.with_name(dest.stem + "_ru.wav")
    to_wav(tts_mp3, ru_wav)
    ru_dur = probe_duration(ru_wav)
    en_start = max(segment.start, window_start)
    en_end = min(segment.end, window_end)
    en_dur = max(en_end - en_start, 0.05)
    en_raw = dest.with_name(dest.stem + "_en.wav")
    slice_wav(source_wav, en_raw, start=en_start - window_start, duration=en_dur)
    en_quiet = dest.with_name(dest.stem + "_en_duck.wav")
    duck_wav(en_raw, en_quiet, duck_db=duck_db)

    if ru_dur <= en_dur + 0.05:
        en_cut = dest.with_name(dest.stem + "_en_cut.wav")
        trim_wav(en_quiet, en_cut, duration=ru_dur)
        mix_wav(ru_wav, en_cut, dest)
        return

    ru_head = dest.with_name(dest.stem + "_ru_head.wav")
    ru_tail = dest.with_name(dest.stem + "_ru_tail.wav")
    overlap = dest.with_name(dest.stem + "_overlap.wav")
    trim_wav(ru_wav, ru_head, duration=en_dur)
    skip_wav(ru_wav, ru_tail, start=en_dur)
    mix_wav(ru_head, en_quiet, overlap)
    concat_wavs([overlap, ru_tail], dest)
