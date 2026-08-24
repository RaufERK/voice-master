from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from . import Glossary
from .glossary import glossary_prompt_block


class TranslationError(RuntimeError):
    pass


class TranslatedItem(BaseModel):
    id: int
    text: str = Field(min_length=1)


class TranslationBatch(BaseModel):
    translations: list[TranslatedItem]


SYSTEM_PROMPT = """You are a professional translator of spoken spiritual lectures.

Translate English speech into natural, spoken Russian.
Keep each item separate. Never merge, split, drop, or add items.
Preserve meaning and a solemn, elevated lecture tone — not slang, not a sermon parody.
Fix obvious speech-to-text errors (wrong names, broken words) without rewriting the sense.
Do not translate timestamps. Do not explain. Return only structured translations.
Use the glossary when a name or term appears.
"""


class OpenAITranslator:
    def __init__(self, *, api_key: str, base_url: str, model: str) -> None:
        from openai import OpenAI

        if not api_key:
            raise TranslationError("OPENAI_API_KEY is missing")
        self.model = model
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url.rstrip("/"),
            timeout=600.0,
            max_retries=2,
        )

    def translate_batch(
        self,
        items: list[dict[str, Any]],
        *,
        previous: list[dict[str, Any]],
        glossary: Glossary | None,
    ) -> list[dict[str, Any]]:
        user_payload = {
            "previous_translated_context": previous,
            "translate": items,
        }
        messages = [
            {"role": "system", "content": _build_system_prompt(glossary)},
            {
                "role": "user",
                "content": (
                    "Translate only the items in 'translate'. "
                    "Return JSON with a 'translations' array of {id, text}. "
                    "Keep the same id for each item.\n\n"
                    + _to_json(user_payload)
                ),
            },
        ]
        try:
            completion = self.client.chat.completions.parse(
                model=self.model,
                messages=messages,
                response_format=TranslationBatch,
            )
            message = completion.choices[0].message
            parsed = message.parsed
            if parsed is None:
                refusal = getattr(message, "refusal", None)
                raise TranslationError(
                    f"OpenAI returned no structured translation: {refusal or message.content}"
                )
            return [{"id": item.id, "text": item.text} for item in parsed.translations]
        except TranslationError:
            raise
        except Exception as exc:
            fallback = self._parse_json_fallback(messages)
            if fallback is not None:
                return fallback
            raise TranslationError(f"OpenAI request failed: {exc}") from exc

    def _parse_json_fallback(self, messages: list[dict[str, str]]) -> list[dict[str, Any]] | None:
        import json
        import re

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
            )
            content = completion.choices[0].message.content or ""
            match = re.search(r"\{[\s\S]*\}", content)
            payload = json.loads(match.group(0) if match else content)
            parsed = TranslationBatch.model_validate(payload)
            return [{"id": item.id, "text": item.text} for item in parsed.translations]
        except Exception:
            return None


def _build_system_prompt(glossary: Glossary | None) -> str:
    block = glossary_prompt_block(glossary)
    if not block:
        return SYSTEM_PROMPT
    return SYSTEM_PROMPT + "\n\nGlossary and context:\n" + block


def _to_json(payload: dict[str, Any]) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False, indent=2)
