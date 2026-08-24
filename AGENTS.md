# voice-master — agent notes

Local Python CLI: turn an English lecture (YouTube SBV + source audio) into **Russian speech audio**. Video is a source only. Do not remux, freeze, or rewrite the MP4.

## Goal (do not drift)

End state is an audio file. Ultimate target: the original lecturer speaks the Russian text **in their own timbre**. Reach that in stages; do not start with cloning.

## Pipeline

```
MP4 audio + captions.sbv
  → parse/merge cues, mark speech vs song
  → clean English ASR (speech only)
  → translate via OpenAI proxy
  → TTS via OpenAI proxy
  → mix:
       stage 1: Russian speech only (songs passthrough)
       stage 2: quiet English bed + pause EN when RU is longer
  → later: same mix, cloned lecturer voice instead of stock TTS
```

## OpenAI via proxy

Same setup as `mkv-ru-subs`. Never call `api.openai.com` from Russia.

- `OPENAI_BASE_URL=https://spoken-word.info/openai-proxy/v1`
- `OPENAI_API_KEY` on the client is a **proxy token**, not `sk-proj-`
- Default translation model: `gpt-5.6-luna`
- TTS: `client.audio.speech` through the same base URL
- Do not log secrets

Proxy code: `amster/openai-proxy`. Current named clients are `pearls` and `mkv-ru-subs`. This project may reuse an existing client token, or get its own token added to the proxy later. A real OpenAI key 401s on the proxy.

## Stack

Python 3.12 venv, `openai`, `pydantic`, `python-dotenv`, system `ffmpeg` / `ffprobe`. No web app.

- From repo root: `.venv/bin/python make_ru_voice.py --check`
- First working pass: `.venv/bin/python make_ru_voice.py --duration 300` (translate + TTS, no English bed)


## Layout

```
voice-master/
  glossaries/               # spiritual names/terms
  audio-source/             # gitignored: per-lecture mp4 + captions
    LECTURE_1/
  work/                     # gitignored checkpoints
  output/                   # gitignored wav/mp3
```

Do not commit `.env`, `.venv`, `audio-source/`, `work/`, `output/`, media, or captions (`*.sbv`, `*.srt`, …).

## Constraints

- User-facing replies: Russian. Code/identifiers: English.
- Do not git-commit or push unless the user asks.
- Stage 1 before English-bed mix. Clone only after a working Russian audio file exists.
- Songs/hymns: keep original audio, do not translate or TTS them.
