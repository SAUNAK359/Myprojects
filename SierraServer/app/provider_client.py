import base64
import io
import json
import time
from typing import Any

import httpx
from PIL import Image, ImageOps

from app.config import settings
from app.models import Provider
from app.provider_crypto import decrypt_secret
from app.schemas import SierraPlan


PHOTOGRAPHY_PROMPT = """
You are Sierra AI, an elite computational-photography director controlling a real Android camera through a constrained plan.
Study subject, lighting, skin, motion, depth, background, highlight risk, shadow detail, composition, mood and probable user intent as a professional photographer would.
The Android client runs Sierra beta locally. Sierra beta is the execution and safety authority: it validates your plan against the real camera and may reject unsafe or unsupported actions.
Never invent a camera control, tool ID, range, lens, extension, or capability that is not explicitly present in DEVICE CAPABILITIES / SKILL CONTEXT.
Use LOCAL SIERRA BETA QUALITY as measured evidence, not as a replacement for visual reasoning.
Choose a deliberate aspect ratio and crop center for the photograph, not merely the sensor ratio.
Aim for an immediately post-ready photograph: technically clean, strong subject placement, natural skin, protected highlights, intentional color and tasteful finish.
Return ONLY one JSON object with these keys:
scene, objective, branch, reference_score, reference_accept, calls, recapture, aspect_ratio, crop_center_x, crop_center_y, planner.
Allowed branch: general, portrait, landscape, low_light, motion, group, food, architecture, macro, selfie.
Allowed aspect_ratio: 4:3, 3:2, 4:5, 1:1, 16:9, 9:16, 2.39:1.
Each calls item has exactly id,v,x,y,s. Use at most 8 calls. Scores and crop centers are 0..1.
If uncertain, prefer fewer safe calls and a natural finish. planner must be sierra_ai_server.
""".strip()


def prepare_image(raw: bytes) -> str:
    with Image.open(io.BytesIO(raw)) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail(
            (settings.max_image_dimension, settings.max_image_dimension),
            Image.Resampling.LANCZOS,
        )
        out = io.BytesIO()
        image.save(out, format="JPEG", quality=74, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(out.getvalue()).decode()


def build_prompt(
    *,
    camera_context_json: str | None,
    device_capabilities_json: str | None,
    local_quality_json: str | None,
    skill_context: str | None,
) -> str:
    return "\n\n".join(
        [
            PHOTOGRAPHY_PROMPT,
            "DEVICE CAPABILITIES:\n" + (device_capabilities_json or "{}"),
            "CURRENT CAMERA STATE:\n" + (camera_context_json or "{}"),
            "LOCAL SIERRA BETA QUALITY:\n" + (local_quality_json or "{}"),
            "SELECTED PHOTOGRAPHY SKILL CONTEXT:\n" + (skill_context or "general photography"),
        ]
    )


def _extract_content(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        raise ValueError("provider returned no choices")
    message = choices[0].get("message") or {}
    parsed = message.get("parsed")
    if isinstance(parsed, dict):
        return json.dumps(parsed)
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                for key in ("text", "content", "value"):
                    if isinstance(part.get(key), str):
                        parts.append(part[key])
                        break
        return "\n".join(parts).strip()
    raise ValueError("provider returned empty content")


def _json_objects(text: str):
    depth = 0
    start = None
    quoted = False
    escaped = False
    for index, char in enumerate(text):
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if char == '"':
            quoted = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                yield text[start : index + 1]
                start = None


def parse_plan(text: str) -> SierraPlan:
    cleaned = text.strip()
    candidates = [cleaned, *_json_objects(cleaned)]
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                for key in ("plan", "result", "analysis", "response", "data"):
                    if isinstance(value.get(key), dict):
                        value = value[key]
                        break
                if "scene" in value and "objective" in value:
                    value.setdefault("planner", "sierra_ai_server")
                    value.setdefault("aspect_ratio", "4:3")
                    value.setdefault("crop_center_x", 0.5)
                    value.setdefault("crop_center_y", 0.5)
                    return SierraPlan.model_validate(value)
        except Exception:
            continue
    raise ValueError("provider did not return a valid Sierra plan")


async def call_provider(
    provider: Provider,
    image_bytes: bytes,
    *,
    camera_context_json: str | None = None,
    device_capabilities_json: str | None = None,
    local_quality_json: str | None = None,
    skill_context: str | None = None,
) -> tuple[SierraPlan, int]:
    image_url = prepare_image(image_bytes)
    prompt = build_prompt(
        camera_context_json=camera_context_json,
        device_capabilities_json=device_capabilities_json,
        local_quality_json=local_quality_json,
        skill_context=skill_context,
    )
    body: dict[str, Any] = {
        "model": provider.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        "temperature": 0.25,
        "top_p": 0.9,
        "max_tokens": provider.max_tokens,
        "stream": False,
    }
    # Generic OpenAI-compatible providers are not required to implement response_format.
    # Only routes explicitly marked by the administrator receive JSON Schema; every other
    # route is constrained by the photography prompt and normalized by parse_plan().
    if provider.supports_json_schema:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "sierra_plan",
                "strict": False,
                "schema": SierraPlan.model_json_schema(),
            },
        }

    started = time.monotonic()
    timeout = httpx.Timeout(provider.timeout_ms / 1000.0)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
        response = await client.post(
            provider.endpoint,
            headers={
                "Authorization": f"Bearer {decrypt_secret(provider.secret_ciphertext)}",
                "Content-Type": "application/json",
                "X-Title": "Sierra Vision",
            },
            json=body,
        )
    latency = int((time.monotonic() - started) * 1000)
    if response.status_code >= 400:
        detail = response.text.replace("\n", " ")[:220]
        raise RuntimeError(f"provider HTTP {response.status_code}: {detail}")
    payload = response.json()
    return parse_plan(_extract_content(payload)), latency
