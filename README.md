# voice-master

Локальный CLI: английская лекция → **русская озвучка (аудио)**.  
Видео не трогаем: MP4 только как источник звука и (позже) образца голоса.

Конечная цель — диктор говорит этот текст по-русски **своим тембром**. Сначала простой перевод + TTS.

В git — только код. Исходные видео/аудио, субтитры и результаты озвучки в репозиторий не входят.

## Стек

Python 3.12, OpenAI SDK, системный `ffmpeg`.  
Запросы идут в Амстердам-прокси, не в `api.openai.com` — тот же способ, что в `mkv-ru-subs` (из России так и задумано).

## Этапы

1. Перевод субтитров + русская озвучка, **без** английского фона. Песни оставляем оригиналом.
2. Тот же файл, но тихий английский под речью; если русский длиннее — пауза EN.
3. Клон голоса лектора вместо обычного TTS.

Подробности: [`PLAN.md`](PLAN.md).

## Быстрый старт

```bash
cd /Users/rauf/Documents/WEB_PROJ/local/voice-master
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env   # OPENAI_API_KEY = токен прокси, не sk-proj
```

Исходники кладём локально, например:

```
audio-source/LECTURE_1/captions.sbv
audio-source/LECTURE_1/*.mp4
```

```bash
.venv/bin/python make_ru_voice.py --check

# этап 1: перевод + русская озвучка, первые 5 минут
.venv/bin/python make_ru_voice.py --duration 300
```

## Окружение

```bash
OPENAI_BASE_URL=https://spoken-word.info/openai-proxy/v1
OPENAI_API_KEY=          # токен прокси
TRANSLATION_MODEL=gpt-5.6-luna
TTS_MODEL=tts-1-hd
```

Артефакты пайплайна — `work/` и `output/` (тоже не в git).
