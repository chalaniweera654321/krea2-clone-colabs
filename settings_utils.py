"""Portable settings and PNG metadata helpers for the Krea 2 Space."""

from __future__ import annotations

import json
import hashlib
import math
import pathlib
import re
from pathlib import Path
from typing import Any

from PIL import Image, PngImagePlugin


APP_ID = "krea-2-turbo-i2i"
PROFILE_SCHEMA_VERSION = 3
CUSTOM_LORA_EXTENSIONS = {".safetensors", ".pt", ".ckpt", ".bin"}
CUSTOM_BASE_MODEL_EXTENSIONS = {".safetensors", ".pt", ".ckpt", ".bin"}
_HF_REPO_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$"
)


def _finite_number(value: Any, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _text(value: Any, field: str, limit: int = 8192) -> str:
    result = "" if value is None else str(value)
    if "\x00" in result or len(result) > limit:
        raise ValueError(f"{field} is invalid or too long")
    return result.strip()


def _finite_number(value: Any, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def validate_custom_lora(row: Any) -> dict[str, Any]:
    """Validate and normalize one custom Hugging Face LoRA row."""
    if isinstance(row, dict):
        repo_id = row.get("repo_id", "")
        filename = row.get("filename", row.get("file", ""))
        revision = row.get("revision", "")
        weight = row.get("weight", 0.0)
    elif isinstance(row, (list, tuple)):
        values = list(row) + [""] * 4
        repo_id, filename, revision, weight = values[:4]
    else:
        raise ValueError("custom LoRA rows must be objects or four-column arrays")

    repo_id = _text(repo_id, "custom LoRA repository", 193)
    if not _HF_REPO_RE.fullmatch(repo_id):
        raise ValueError(
            f"invalid Hugging Face repository ID {repo_id!r}; expected namespace/name"
        )

    filename = _text(filename, "custom LoRA file", 512).replace("\\", "/")
    path = pathlib.PurePosixPath(filename)
    if (
        not filename
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.suffix.lower() not in CUSTOM_LORA_EXTENSIONS
    ):
        allowed = ", ".join(sorted(CUSTOM_LORA_EXTENSIONS))
        raise ValueError(f"custom LoRA file must be relative and end in {allowed}")

    revision = _text(revision, "custom LoRA revision", 256)
    if any(ord(char) < 32 for char in revision):
        raise ValueError("custom LoRA revision contains control characters")

    weight_number = _finite_number(weight or 0.0, "custom LoRA weight")
    if weight_number < -3.0 or weight_number > 3.0:
        raise ValueError("custom LoRA weight must be between -3 and 3")

    return {
        "repo_id": repo_id,
        "filename": filename,
        "revision": revision,
        "weight": round(weight_number, 6),
    }


def normalize_custom_loras(rows: Any) -> tuple[list[dict[str, Any]], list[str]]:
    """Normalize non-empty custom rows and collect row errors as warnings."""
    if rows is None:
        return [], []
    if isinstance(rows, dict):
        rows = [rows]
    normalized: list[dict[str, Any]] = []
    warnings: list[str] = []
    for index, row in enumerate(rows):
        if row is None or row == [] or row == {}:
            continue
        if isinstance(row, (list, tuple)):
            values = list(row) + [""] * 4
            if not any(str(value).strip() for value in values[:3]) and not values[3]:
                continue
        try:
            normalized.append(validate_custom_lora(row))
        except ValueError as exc:
            warnings.append(f"custom LoRA row {index + 1}: {exc}")
    return normalized, warnings


def validate_custom_base_model(value: Any) -> dict[str, str]:
    """Validate one custom Hugging Face diffusion-model reference."""
    if isinstance(value, dict):
        repo_id = value.get("repo_id", "")
        filename = value.get("filename", value.get("file", ""))
        revision = value.get("revision", "")
    elif isinstance(value, (list, tuple)):
        values = list(value) + [""] * 3
        repo_id, filename, revision = values[:3]
    else:
        raise ValueError("custom base model must be an object or three-column array")

    repo_id = _text(repo_id, "custom base-model repository", 193)
    if not _HF_REPO_RE.fullmatch(repo_id):
        raise ValueError(
            f"invalid Hugging Face repository ID {repo_id!r}; expected namespace/name"
        )

    filename = _text(filename, "custom base-model file", 512).replace("\\", "/")
    path = pathlib.PurePosixPath(filename)
    if (
        not filename
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.suffix.lower() not in CUSTOM_BASE_MODEL_EXTENSIONS
    ):
        allowed = ", ".join(sorted(CUSTOM_BASE_MODEL_EXTENSIONS))
        raise ValueError(f"custom base-model file must be relative and end in {allowed}")

    revision = _text(revision, "custom base-model revision", 256)
    if any(ord(char) < 32 for char in revision):
        raise ValueError("custom base-model revision contains control characters")
    return {"repo_id": repo_id, "filename": filename, "revision": revision}


def stable_custom_lora_namespace(repo_id: str, filename: str, revision: str = "") -> str:
    """Return a filesystem-safe stable namespace for one remote LoRA."""
    identity = "\x00".join((repo_id, revision, filename)).encode("utf-8")
    return hashlib.sha256(identity).hexdigest()[:20]


def stable_custom_base_model_namespace(repo_id: str, filename: str, revision: str = "") -> str:
    """Return a filesystem-safe stable namespace for one remote base model."""
    identity = "\x00".join((repo_id, revision, filename)).encode("utf-8")
    return hashlib.sha256(identity).hexdigest()[:20]


def build_settings(
    *,
    mode: str,
    prompt: str,
    edit_prompt: str,
    width: int,
    height: int,
    target_megapixels: float,
    grounding_px: int,
    ref_boost: float,
    ref_boost_a: float,
    steps: int,
    cfg: float,
    sampler_name: str,
    scheduler: str,
    seed: int,
    randomize_seed: bool,
    gen_budget: float,
    effective_seed: int | None = None,
    base_model: str = "",
    custom_base_model: dict[str, Any] | None = None,
    catalog_loras: list[dict[str, Any]] | None = None,
    custom_loras: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the canonical profile embedded in generated PNG files."""
    return {
        "app": APP_ID,
        "schema_version": PROFILE_SCHEMA_VERSION,
        "base_model": _text(base_model, "base model", 256),
        "custom_base_model": (
            validate_custom_base_model(custom_base_model)
            if custom_base_model
            else None
        ),
        "mode": _text(mode, "mode", 32),
        "prompt": _text(prompt, "prompt"),
        "edit_prompt": _text(edit_prompt, "edit prompt"),
        "width": int(width),
        "height": int(height),
        "target_megapixels": float(target_megapixels),
        "grounding_px": int(grounding_px),
        "ref_boost": float(ref_boost),
        "ref_boost_a": float(ref_boost_a),
        "steps": int(steps),
        "cfg": float(cfg),
        "sampler_name": _text(sampler_name, "sampler", 64),
        "scheduler": _text(scheduler, "scheduler", 64),
        "seed": int(seed),
        "effective_seed": None if effective_seed is None else int(effective_seed),
        "randomize_seed": bool(randomize_seed),
        "gen_budget": float(gen_budget),
        "catalog_loras": [
            {
                "hf_filename": _text(item.get("hf_filename", ""), "catalog LoRA filename", 512),
                "weight": float(item.get("weight", 0.0)),
            }
            for item in (catalog_loras or [])
            if item.get("hf_filename") and abs(float(item.get("weight", 0.0))) > 1e-6
        ],
        "custom_loras": [validate_custom_lora(item) for item in (custom_loras or [])],
    }


def parse_settings_text(text: Any) -> tuple[dict[str, Any], list[str]]:
    """Parse a Krea JSON profile or a compact A1111-style parameter string."""
    value = "" if text is None else str(text).strip()
    if not value:
        return {}, ["settings text is empty"]
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        lines = value.splitlines()
        result: dict[str, Any] = {}
        steps = re.search(r"Steps:\s*(\d+)", value, re.I)
        cfg = re.search(r"CFG scale:\s*([\d.]+)", value, re.I)
        sampler = re.search(r"Sampler:\s*([^,\n]+)", value, re.I)
        seed = re.search(r"Seed:\s*(\d+)", value, re.I)
        size = re.search(r"Size:\s*(\d+)\s*[xX×]\s*(\d+)", value, re.I)
        negative_line = next(
            (index for index, line in enumerate(lines) if line.lower().startswith("negative prompt:")),
            None,
        )
        if negative_line is not None:
            result["prompt"] = "\n".join(lines[:negative_line]).strip()
        else:
            result["prompt"] = lines[0].strip() if lines else ""
        if steps:
            result["steps"] = int(steps.group(1))
        if cfg:
            result["cfg"] = float(cfg.group(1))
        if sampler:
            result["sampler_name"] = sampler.group(1).strip()
        if seed:
            result["effective_seed"] = int(seed.group(1))
        if size:
            result["width"] = int(size.group(1))
            result["height"] = int(size.group(2))
        return result, []
    if not isinstance(decoded, dict):
        return {}, ["settings JSON must be an object"]
    if isinstance(decoded.get("krea_settings"), str):
        try:
            decoded = json.loads(decoded["krea_settings"])
        except json.JSONDecodeError:
            return {}, ["krea_settings metadata is not valid JSON"]
    return decoded, []


def extract_image_settings(path: str | Path) -> tuple[dict[str, Any], list[str]]:
    """Extract Krea settings or common parameter text from an image."""
    try:
        with Image.open(path) as image:
            metadata = dict(image.info)
    except Exception as exc:
        return {}, [f"could not read image metadata: {exc}"]
    for key in ("krea_settings", "parameters", "prompt"):
        raw = metadata.get(key)
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="ignore")
        if isinstance(raw, str) and raw.strip():
            data, warnings = parse_settings_text(raw)
            if data:
                return data, warnings
    return {}, ["image contains no recognized Krea settings"]


def build_parameters_text(settings: dict[str, Any]) -> str:
    """Build a readable generation summary for image viewers."""
    prompt = settings.get("edit_prompt") or settings.get("prompt", "")
    parts = [
        f"Base model: {settings.get('base_model', '')}",
        f"Steps: {settings.get('steps')}",
        f"CFG scale: {settings.get('cfg')}",
        f"Sampler: {settings.get('sampler_name')}",
        f"Schedule type: {settings.get('scheduler')}",
        f"Seed: {settings.get('effective_seed', settings.get('seed'))}",
        f"Size: {settings.get('width')}x{settings.get('height')}",
    ]
    if settings.get("mode") == "edit":
        parts.extend([
            f"Grounding: {settings.get('grounding_px')}px",
            f"Reference strength: {settings.get('ref_boost')}",
            f"Second reference strength: {settings.get('ref_boost_a')}",
        ])
    lora_tags = []
    for item in settings.get("catalog_loras", []):
        name = pathlib.PurePosixPath(str(item.get("hf_filename", ""))).stem
        lora_tags.append(f"{name}:{float(item.get('weight', 0.0)):g}")
    for item in settings.get("custom_loras", []):
        name = pathlib.PurePosixPath(str(item.get("filename", ""))).stem
        lora_tags.append(f"{name}:{float(item.get('weight', 0.0)):g}")
    if lora_tags:
        parts.append("LoRAs: " + ", ".join(lora_tags))
    return f"{prompt}\n" + ", ".join(parts)


def write_png_metadata(source: str | Path, destination: str | Path, settings: dict[str, Any]) -> None:
    """Copy an image while adding canonical and readable Krea metadata."""
    with Image.open(source) as image:
        output = image.copy()
        info = PngImagePlugin.PngInfo()
        for key, value in image.info.items():
            if key not in {"krea_settings", "parameters"} and isinstance(value, str):
                info.add_text(key, value)
        info.add_text("krea_settings", json.dumps(settings, sort_keys=True))
        info.add_text("parameters", build_parameters_text(settings))
        output.save(destination, format="PNG", pnginfo=info)
