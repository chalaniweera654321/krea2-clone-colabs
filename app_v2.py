from __future__ import annotations

import asyncio
import gc
import glob
import json
import logging
import os
import pathlib
import random
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download
from PIL import Image

# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    force=True,
)
LOG = logging.getLogger("krea2")

def log(msg: str, *args: Any) -> None:
    LOG.info(msg, *args)

def log_exception(where: str, exc: BaseException) -> None:
    LOG.error("[%s] %s: %s", where, type(exc).__name__, exc)
    LOG.error(traceback.format_exc())

# ============================================================================
# MEGA
# ============================================================================

_MEGA_UPLOAD_LOCK = threading.Lock()
MEGA_ACCOUNT = None

def _mega_login():
    email = os.environ.get("MEGA_EMAIL")
    password = os.environ.get("MEGA_PASSWORD")
    if not email or not password:
        raise RuntimeError(
            "Missing MEGA_EMAIL or MEGA_PASSWORD. "
            "Set MEGA_EMAIL and MEGA_PASSWORD in Colab environment variables."
        )

    try:
        from mega import Mega
    except ImportError as exc:
        raise RuntimeError(
            "MEGA package is missing. Install with: pip install -q mega.py-v2"
        ) from exc

    try:
        log("[mega] logging in")
        account = Mega().login(email, password)
        account.get_files()
        log("[mega] login verification successful")
        return account
    except Exception as exc:
        log_exception("mega-login", exc)
        raise RuntimeError(
            "MEGA login verification failed. Check MEGA_EMAIL and MEGA_PASSWORD."
        ) from exc

def _mega_remote_filenames(account) -> set[str]:
    files = account.get_files() or {}
    names: set[str] = set()
    for node in files.values():
        if not isinstance(node, dict):
            continue
        attrs = node.get("a", {})
        if isinstance(attrs, dict):
            name = attrs.get("n")
            if isinstance(name, str) and name:
                names.add(name)
    return names

def _unique_mega_filename(existing: set[str], extension: str) -> str:
    timestamp = datetime.now().strftime("Image %b %d, %Y, %I_%M_%S %p")
    base = f"{timestamp}{extension}"
    if base not in existing:
        return base
    n = 1
    while True:
        candidate = f"{timestamp}_{n:03d}{extension}"
        if candidate not in existing:
            return candidate
        n += 1

def _upload_to_mega(file_path: str, account) -> str:
    path = pathlib.Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Generated file does not exist: {file_path}")

    with _MEGA_UPLOAD_LOCK:
        log("[mega] checking remote filenames")
        existing = _mega_remote_filenames(account)
        filename = _unique_mega_filename(existing, path.suffix or ".png")
        log("[mega] uploading %s as %s", path, filename)
        uploaded = account.upload(str(path), dest=None, dest_filename=filename)
        log("[mega] uploaded: %s", filename)
        return str(uploaded)

# ============================================================================
# SETTINGS
# ============================================================================
# SETTINGS
# ============================================================================

from settings_utils import (
    build_settings,
    extract_image_settings,
    parse_settings_text,
    write_png_metadata,
)

# ============================================================================
# PATHS
# ============================================================================

ROOT = Path(__file__).resolve().parent

def _detect_comfy_root() -> Path:
    candidates = [
        ROOT,
        ROOT / "ComfyUI",
        Path("/content/ComfyUI"),
    ]
    for candidate in candidates:
        if (candidate / "main.py").is_file() and (candidate / "models").is_dir():
            return candidate
    return Path("/content/ComfyUI")

COMFY = _detect_comfy_root()
MODELS = Path(os.environ.get("KREA_MODELS_DIR", "/content/krea2-models"))

# Import only after paths are known.
import folder_paths

for name in ("diffusion_models", "loras", "vae", "text_encoders"):
    (MODELS / name).mkdir(parents=True, exist_ok=True)
    folder_paths.add_model_folder_path(name, str(MODELS / name))

INPUT = COMFY / "input"
OUTPUT = COMFY / "output"
CUSTOM_NODES = COMFY / "custom_nodes"

T2I_SOURCE = ROOT / "lustifyWorkflowsKrea2_krea2.json"
EDIT_SOURCE = ROOT / "lustifyWorkflowsKrea2_krea2Edit.json"

KREA_EDIT_NODES = "https://github.com/lbouaraba/comfyui-krea2edit.git"

IDENTITY_REPO = "conradlocke/krea2-identity-edit"
IDENTITY_FILE = "krea2_identity_edit_v1_2.safetensors"
IDENTITY_LORA_DIR = MODELS / "loras" / "krea"
IDENTITY_LORA_PATH = IDENTITY_LORA_DIR / IDENTITY_FILE
IDENTITY_COMFY_NAME = pathlib.PurePosixPath("krea", IDENTITY_FILE).as_posix()

TEXT_ENCODER_DIR = MODELS / "text_encoders"
VAE_DIR = MODELS / "vae"
DIFFUSION_DIR = MODELS / "diffusion_models"
LORA_ROOT = MODELS / "loras"

TEXT_ENCODER_FILE = "qwen3vl_4b_fp8_scaled.safetensors"
VAE_FILE = "qwen_image_vae.safetensors"

SAMPLERS = [
    "euler", "euler_ancestral", "euler_a", "dpmpp_2m",
    "dpmpp_2m_sde", "dpmpp_sde", "heun", "lms",
]
SCHEDULERS = [
    "beta", "normal", "karras", "exponential",
    "sgm_uniform", "simple",
]

DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1024
DEFAULT_TARGET_MP = 1.4
MAX_WIDTH = 2048
MAX_HEIGHT = 2048
MAX_TARGET_MP = 4.0
DEFAULT_GROUNDING = 768
DEFAULT_REF_BOOST = 1.0
DEFAULT_STEPS = 8
DEFAULT_CFG = 1.0
DEFAULT_SAMPLER = "euler"
DEFAULT_SCHEDULER = "beta"
DEFAULT_SEED = 2

MIN_GPU_SECONDS = int(os.environ.get("MIN_GPU_SECONDS", "45"))
MAX_GPU_SECONDS = int(os.environ.get("MAX_GPU_SECONDS", "300"))

LOCAL_BASE_MODELS: list[str] = []
LOCAL_LORAS: list[str] = []

# ============================================================================
# PERSISTENT COMFY RUNTIME
# ============================================================================

_COMFY_READY = False
_NODES_READY = False

_COMFY_LOOP: asyncio.AbstractEventLoop | None = None
_COMFY_SERVER = None
_COMFY_EXECUTOR = None
_COMFY_RUNTIME_LOCK = threading.RLock()
_GENERATION_LOCK = threading.Lock()

_WORKFLOW_CACHE: dict[str, dict[str, Any]] = {}

def _memory_snapshot(label: str) -> None:
    try:
        import psutil
        vm = psutil.virtual_memory()
        log(
            "[system] %s | RAM: %.1f%% used | %.2f / %.2f GB",
            label,
            vm.percent,
            vm.used / 1024**3,
            vm.total / 1024**3,
        )
    except Exception as exc:
        log("[system] RAM read failed: %s: %s", type(exc).__name__, exc)

    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            log(
                "[system] GPU: %s | allocated %.2f GB | reserved %.2f GB | "
                "free %.2f / %.2f GB",
                torch.cuda.get_device_name(0),
                allocated / 1024**3,
                reserved / 1024**3,
                free / 1024**3,
                total / 1024**3,
            )
    except Exception as exc:
        log("[system] GPU read failed: %s: %s", type(exc).__name__, exc)

def _cleanup_memory(reason: str = "generation-end") -> None:
    log("[memory] cleanup start: %s", reason)
    _memory_snapshot("before-cleanup")

    try:
        gc.collect()
    except Exception as exc:
        log("[memory] gc.collect failed: %s: %s", type(exc).__name__, exc)

    try:
        import torch
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception as exc:
        log("[memory] CUDA cleanup failed: %s: %s", type(exc).__name__, exc)

    try:
        gc.collect()
    except Exception:
        pass

    _memory_snapshot("after-cleanup")
    log("[memory] cleanup complete: %s", reason)

# ============================================================================
# COMMAND HELPERS
# ============================================================================

def _run(command: list[str], cwd: Path | None = None, check: bool = True) -> None:
    log("[setup] %s", " ".join(command))
    subprocess.run(command, cwd=str(cwd) if cwd else None, check=check)

def _pip_install(arguments: list[str]) -> None:
    _run([sys.executable, "-m", "pip", "install", "--no-cache-dir", *arguments], check=False)

def _install_filtered_requirements(path: Path) -> None:
    if not path.exists():
        log("[requirements] not found: %s", path)
        return

    blocked = {
        "torch", "torchvision", "torchaudio",
        "transformers", "huggingface-hub", "accelerate",
    }
    requirements: list[str] = []

    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        item = raw.strip()
        if not item or item.startswith("#"):
            continue
        package = re.split(r"[<>=!~;\[\s]", item.lower().replace("_", "-"), maxsplit=1)[0]
        if package not in blocked:
            requirements.append(item)

    if requirements:
        _pip_install(requirements)

def _ensure_repo(path: Path, url: str) -> None:
    if path.exists():
        log("[setup] custom node already exists: %s", path)
        return
    _run(["git", "clone", "--depth", "1", url, str(path)])

# ============================================================================
# COMPATIBILITY / DIRECTORIES
# ============================================================================

def _restore_utils_namespace() -> None:
    source = COMFY / "utils"
    target = COMFY / "utilities"

    if not source.exists() and target.exists():
        log("[compat] restoring utilities -> utils")
        target.rename(source)

    if not source.exists():
        return

    for path in COMFY.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        updated = re.sub(r"\bfrom utilities\b", "from utils", text)
        updated = re.sub(r"\bimport utilities\b", "import utils", updated)

        if updated != text:
            path.write_text(updated, encoding="utf-8")

def _ensure_model_directories() -> None:
    for folder in [
        "diffusion_models", "text_encoders", "vae", "loras", "loras/krea"
    ]:
        (MODELS / folder).mkdir(parents=True, exist_ok=True)

    INPUT.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    CUSTOM_NODES.mkdir(parents=True, exist_ok=True)

# ============================================================================
# IDENTITY MODEL / SCANNING
# ============================================================================

def _download_identity_model() -> None:
    IDENTITY_LORA_DIR.mkdir(parents=True, exist_ok=True)

    if IDENTITY_LORA_PATH.exists():
        log("[identity] already installed: %s", IDENTITY_LORA_PATH)
        return

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    log("[identity] downloading %s", IDENTITY_FILE)

    downloaded = Path(
        hf_hub_download(
            repo_id=IDENTITY_REPO,
            filename=IDENTITY_FILE,
            local_dir=str(IDENTITY_LORA_DIR),
            token=token,
        )
    )

    if downloaded.resolve() != IDENTITY_LORA_PATH.resolve():
        shutil.move(str(downloaded), str(IDENTITY_LORA_PATH))

    log("[identity] ready: %s", IDENTITY_LORA_PATH)

def _scan_local_models() -> None:
    global LOCAL_BASE_MODELS, LOCAL_LORAS

    _ensure_model_directories()
    extensions = {".safetensors", ".ckpt", ".pt", ".bin"}

    models: list[str] = []
    if DIFFUSION_DIR.exists():
        for path in DIFFUSION_DIR.rglob("*"):
            if path.is_file() and path.suffix.lower() in extensions:
                rel = path.relative_to(DIFFUSION_DIR)
                models.append(pathlib.PurePosixPath(*rel.parts).as_posix())

    loras: list[str] = []
    if LORA_ROOT.exists():
        for path in LORA_ROOT.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in extensions:
                continue
            try:
                if path.resolve() == IDENTITY_LORA_PATH.resolve():
                    continue
            except OSError:
                pass
            rel = path.relative_to(LORA_ROOT)
            loras.append(pathlib.PurePosixPath(*rel.parts).as_posix())

    LOCAL_BASE_MODELS = sorted(models, key=str.lower)
    LOCAL_LORAS = sorted(loras, key=str.lower)

    log("[models] found %d local diffusion model(s)", len(LOCAL_BASE_MODELS))
    for x in LOCAL_BASE_MODELS:
        log("[models]   %s", x)

    log("[loras] found %d local LoRA(s)", len(LOCAL_LORAS))
    for x in LOCAL_LORAS:
        log("[loras]   %s", x)

def _validate_required_assets() -> None:
    missing = []

    if not (TEXT_ENCODER_DIR / TEXT_ENCODER_FILE).is_file():
        missing.append(f"text encoder: {TEXT_ENCODER_DIR / TEXT_ENCODER_FILE}")
    if not (VAE_DIR / VAE_FILE).is_file():
        missing.append(f"VAE: {VAE_DIR / VAE_FILE}")
    if not IDENTITY_LORA_PATH.is_file():
        missing.append(f"identity adapter: {IDENTITY_LORA_PATH}")

    if missing:
        raise RuntimeError("Missing required local model files:\n" + "\n".join(f"  - {x}" for x in missing))

# ============================================================================
# COMFY INITIALIZATION - ONLY ONCE
# ============================================================================

def _ensure_comfy() -> None:
    global _COMFY_READY

    with _COMFY_RUNTIME_LOCK:
        if _COMFY_READY:
            return

        log("[comfy] using: %s", COMFY)
        if not (COMFY / "main.py").exists():
            raise RuntimeError(f"ComfyUI was not found at: {COMFY}")

        _ensure_model_directories()
        _install_filtered_requirements(COMFY / "requirements.txt")
        _ensure_repo(CUSTOM_NODES / "comfyui-krea2edit", KREA_EDIT_NODES)
        _restore_utils_namespace()
        _download_identity_model()
        _scan_local_models()

        _COMFY_READY = True
        log("[comfy] base setup ready")

def _init_comfy_nodes() -> None:
    global _NODES_READY, _COMFY_LOOP, _COMFY_SERVER, _COMFY_EXECUTOR

    with _COMFY_RUNTIME_LOCK:
        if _NODES_READY:
            return

        comfy_path = str(COMFY)
        sys.path = [x for x in sys.path if x != comfy_path]
        sys.path.insert(0, comfy_path)

        for name in list(sys.modules):
            if name == "utils" or name.startswith("utils."):
                del sys.modules[name]

        os.chdir(COMFY)

        log("[comfy-runtime] importing execution/nodes/server")
        import execution
        import nodes
        import server
        from app.assets.manager import default_asset_manager

        # IMPORTANT:
        # One event loop + one PromptServer + one executor for the lifetime
        # of this Python process. Do not recreate them on every Generate click.
        _COMFY_LOOP = asyncio.new_event_loop()
        asyncio.set_event_loop(_COMFY_LOOP)

        _COMFY_SERVER = server.PromptServer(
            _COMFY_LOOP,
            default_asset_manager(),
        )

        log("[comfy-runtime] initializing extra nodes")
        _COMFY_LOOP.run_until_complete(nodes.init_extra_nodes())

        # Conservative cache for a 12-13 GB RAM Colab runtime.
        _COMFY_EXECUTOR = execution.PromptExecutor(
            _COMFY_SERVER,
            cache_type=execution.CacheType.RAM_PRESSURE,
            cache_args={
                "lru": 0,
                "ram": 0.5,
                "ram_inactive": 2.0,
            },
        )

        _NODES_READY = True
        log("[comfy-runtime] persistent executor ready")
        _memory_snapshot("runtime-ready")

# ============================================================================
# VALIDATION
# ============================================================================

def _validate_model_name(model_name: str) -> str:
    normalized = str(model_name).replace("\\", "/")
    path = pathlib.PurePosixPath(normalized)

    if (
        not normalized or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("invalid diffusion model path")

    candidate = DIFFUSION_DIR / Path(*path.parts)
    try:
        candidate.resolve().relative_to(DIFFUSION_DIR.resolve())
    except ValueError as exc:
        raise ValueError("invalid diffusion model path") from exc

    if not candidate.is_file():
        raise ValueError(f"diffusion model is not installed: {normalized}")

    return path.as_posix()

def _validate_lora_name(lora_name: str) -> str:
    normalized = str(lora_name).replace("\\", "/")
    path = pathlib.PurePosixPath(normalized)

    if (
        not normalized or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("invalid LoRA path")

    candidate = LORA_ROOT / Path(*path.parts)
    try:
        candidate.resolve().relative_to(LORA_ROOT.resolve())
    except ValueError as exc:
        raise ValueError("invalid LoRA path") from exc

    if not candidate.is_file():
        raise ValueError(f"LoRA is not installed: {normalized}")

    if candidate.resolve() == IDENTITY_LORA_PATH.resolve():
        raise ValueError("identity adapter cannot be selected as a user LoRA")

    return path.as_posix()

# ============================================================================
# WORKFLOWS
# ============================================================================

def _read_source_workflow(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"workflow file is missing: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not data.get("nodes"):
        raise ValueError(f"workflow file has no nodes: {path.name}")
    return data

def _ref(node: str, output: int = 0) -> list[Any]:
    return [node, output]

def _t2i_workflow(base_model: str) -> dict[str, Any]:
    base_model = _validate_model_name(base_model)
    key = f"text2image:{base_model}"

    if key in _WORKFLOW_CACHE:
        return json.loads(json.dumps(_WORKFLOW_CACHE[key]))

    _read_source_workflow(T2I_SOURCE)

    workflow = {
        "1": {"class_type": "UNETLoader", "inputs": {
            "unet_name": base_model, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {
            "clip_name": TEXT_ENCODER_FILE, "type": "krea2", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VAE_FILE}},
        "4": {"class_type": "ModelSamplingAuraFlow", "inputs": {
            "model": _ref("1"), "shift": 4.0}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {
            "clip": _ref("2"), "text": ""}},
        "6": {"class_type": "ConditioningZeroOut", "inputs": {
            "conditioning": _ref("5")}},
        "7": {"class_type": "EmptyLatentImage", "inputs": {
            "width": DEFAULT_WIDTH, "height": DEFAULT_HEIGHT, "batch_size": 1}},
        "8": {"class_type": "KSampler", "inputs": {
            "model": _ref("4"), "positive": _ref("5"), "negative": _ref("6"),
            "latent_image": _ref("7"), "seed": DEFAULT_SEED,
            "steps": DEFAULT_STEPS, "cfg": DEFAULT_CFG,
            "sampler_name": DEFAULT_SAMPLER, "scheduler": DEFAULT_SCHEDULER,
            "denoise": 1.0}},
        "9": {"class_type": "VAEDecode", "inputs": {
            "samples": _ref("8"), "vae": _ref("3")}},
        "10": {"class_type": "SaveImage", "inputs": {
            "images": _ref("9"), "filename_prefix": "krea2_turbo"}},
    }

    _WORKFLOW_CACHE[key] = workflow
    return json.loads(json.dumps(workflow))

def _edit_workflow(has_second_reference: bool, base_model: str) -> dict[str, Any]:
    base_model = _validate_model_name(base_model)
    _read_source_workflow(EDIT_SOURCE)

    workflow = {
        "1": {"class_type": "LoadImage", "inputs": {"image": ""}},
        "3": {"class_type": "CLIPLoader", "inputs": {
            "clip_name": TEXT_ENCODER_FILE, "type": "krea2", "device": "default"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": VAE_FILE}},
        "5": {"class_type": "UNETLoader", "inputs": {
            "unet_name": base_model, "weight_dtype": "default"}},
        "6": {"class_type": "LoraLoaderModelOnly", "inputs": {
            "model": _ref("5"), "lora_name": IDENTITY_COMFY_NAME,
            "strength_model": 1.0}},
        "7": {"class_type": "VAEEncode", "inputs": {
            "pixels": _ref("1"), "vae": _ref("4")}},
        "8": {"class_type": "EmptySD3LatentImage", "inputs": {
            "width": DEFAULT_WIDTH, "height": DEFAULT_HEIGHT, "batch_size": 1}},
        "9": {"class_type": "Krea2EditModelPatch", "inputs": {
            "model": _ref("6"), "source_latent": _ref("7"), "vae": _ref("4"),
            "source_image": _ref("1"), "target_latent": _ref("8"),
            "ref_boost": DEFAULT_REF_BOOST, "ref_boost_a": DEFAULT_REF_BOOST,
            "fit_mode": "fit"}},
        "10": {"class_type": "Krea2EditGroundedEncode", "inputs": {
            "clip": _ref("3"), "image": _ref("1"), "prompt": "",
            "grounding_px": DEFAULT_GROUNDING}},
        "11": {"class_type": "ConditioningZeroOut", "inputs": {
            "conditioning": _ref("10")}},
        "12": {"class_type": "ModelSamplingAuraFlow", "inputs": {
            "model": _ref("9"), "shift": 4.0}},
        "13": {"class_type": "KSampler", "inputs": {
            "model": _ref("12"), "positive": _ref("10"), "negative": _ref("11"),
            "latent_image": _ref("8"), "seed": DEFAULT_SEED,
            "steps": DEFAULT_STEPS, "cfg": DEFAULT_CFG,
            "sampler_name": DEFAULT_SAMPLER, "scheduler": DEFAULT_SCHEDULER,
            "denoise": 1.0}},
        "14": {"class_type": "VAEDecode", "inputs": {
            "samples": _ref("13"), "vae": _ref("4")}},
        "15": {"class_type": "SaveImage", "inputs": {
            "images": _ref("14"), "filename_prefix": "krea2_edit"}},
    }

    if has_second_reference:
        workflow["2"] = {"class_type": "LoadImage", "inputs": {"image": ""}}
        workflow["16"] = {"class_type": "VAEEncode", "inputs": {
            "pixels": _ref("2"), "vae": _ref("4")}}
        workflow["9"]["inputs"]["source_latent_b"] = _ref("16")
        workflow["9"]["inputs"]["source_image_b"] = _ref("2")
        workflow["10"]["inputs"]["image_b"] = _ref("2")

    return workflow

def _inject_lora_chain(
    workflow: dict[str, Any],
    enabled_loras: list[tuple[str, float]],
    *,
    model_source: list[Any],
    clip_source: list[Any],
    model_consumers: list[tuple[str, str]],
    clip_consumers: list[tuple[str, str]],
) -> None:
    if not enabled_loras:
        return

    previous_model = model_source
    previous_clip = clip_source

    for index, (filename, strength) in enumerate(enabled_loras):
        node_id = f"user_lora_{index}"
        workflow[node_id] = {
            "class_type": "LoraLoader",
            "inputs": {
                "model": previous_model,
                "clip": previous_clip,
                "lora_name": filename,
                "strength_model": float(strength),
                "strength_clip": float(strength),
            },
        }
        previous_model = _ref(node_id)
        previous_clip = _ref(node_id, 1)

    for node_id, input_name in model_consumers:
        workflow[node_id]["inputs"][input_name] = previous_model
    for node_id, input_name in clip_consumers:
        workflow[node_id]["inputs"][input_name] = previous_clip

# ============================================================================
# IMAGE HELPERS
# ============================================================================

def _prepare_edit_image(path: str, target_megapixels: float) -> tuple[str, int, int]:
    with Image.open(path) as source:
        image = source.convert("RGB")

    megapixels = max(0.25, min(MAX_TARGET_MP, float(target_megapixels)))
    scale = (megapixels * 1_000_000 / max(1, image.width * image.height)) ** 0.5

    width = max(64, int(round(image.width * scale / 64) * 64))
    height = max(64, int(round(image.height * scale / 64) * 64))
    width = min(MAX_WIDTH, width)
    height = min(MAX_HEIGHT, height)

    image = image.resize((width, height), Image.Resampling.LANCZOS)
    name = f"input_{uuid.uuid4().hex[:12]}.png"
    image.save(INPUT / name, format="PNG")
    return name, width, height

def _stage_image(path: str, prefix: str) -> str:
    with Image.open(path) as source:
        image = source.convert("RGB")
    name = f"{prefix}_{uuid.uuid4().hex[:12]}.png"
    image.save(INPUT / name, format="PNG")
    return name

# ============================================================================
# INJECTION
# ============================================================================

def _inject_t2i(
    workflow: dict[str, Any],
    *,
    prompt: str, width: int, height: int, steps: int, cfg: float,
    sampler: str, scheduler: str, seed: int,
    enabled_loras: list[tuple[str, float]] | None = None,
) -> None:
    _inject_lora_chain(
        workflow, enabled_loras or [],
        model_source=_ref("1"), clip_source=_ref("2"),
        model_consumers=[("4", "model")],
        clip_consumers=[("5", "clip")],
    )
    workflow["5"]["inputs"]["text"] = prompt.strip()
    workflow["7"]["inputs"].update(width=int(width), height=int(height))
    workflow["8"]["inputs"].update(
        seed=int(seed), steps=int(steps), cfg=float(cfg),
        sampler_name=sampler, scheduler=scheduler, denoise=1.0,
    )

def _inject_edit(
    workflow: dict[str, Any],
    *,
    primary_name: str, second_name: str | None,
    width: int, height: int, edit_prompt: str,
    grounding_px: int, ref_boost: float, ref_boost_a: float,
    steps: int, cfg: float, sampler: str, scheduler: str, seed: int,
    enabled_loras: list[tuple[str, float]] | None = None,
) -> None:
    _inject_lora_chain(
        workflow, enabled_loras or [],
        model_source=_ref("6"), clip_source=_ref("3"),
        model_consumers=[("9", "model")],
        clip_consumers=[("10", "clip")],
    )

    workflow["1"]["inputs"]["image"] = primary_name
    workflow["8"]["inputs"].update(width=int(width), height=int(height))
    workflow["9"]["inputs"].update(
        ref_boost=float(ref_boost), ref_boost_a=float(ref_boost_a)
    )
    workflow["10"]["inputs"].update(
        prompt=edit_prompt.strip(), grounding_px=int(grounding_px)
    )
    workflow["13"]["inputs"].update(
        seed=int(seed), steps=int(steps), cfg=float(cfg),
        sampler_name=sampler, scheduler=scheduler, denoise=1.0,
    )

    if second_name:
        workflow["2"]["inputs"]["image"] = second_name

# ============================================================================
# EXECUTION
# ============================================================================

def _find_node(workflow: dict[str, Any], class_type: str) -> str:
    for node_id, node in workflow.items():
        if node.get("class_type") == class_type:
            return node_id
    raise KeyError(f"workflow does not contain {class_type}")

def _executor_status_watcher(stop_event: threading.Event, started: float) -> None:
    last = -1
    while not stop_event.wait(10):
        elapsed = int(time.time() - started)
        log("[heartbeat] generation still running: %ss", elapsed)
        _memory_snapshot(f"generation-{elapsed}s")

def _execute_workflow(workflow: dict[str, Any]) -> list[str]:
    global _COMFY_EXECUTOR

    if not _NODES_READY or _COMFY_EXECUTOR is None:
        raise RuntimeError("ComfyUI executor is not initialized")

    prompt_id = str(uuid.uuid4())
    save_id = _find_node(workflow, "SaveImage")

    log("[execute] prompt_id=%s", prompt_id)
    log("[execute] output node=%s", save_id)
    log("[execute] nodes: %s", ", ".join(
        f"{k}:{v.get('class_type')}" for k, v in workflow.items()
    ))
    _memory_snapshot("before-executor")

    stop_event = threading.Event()
    watcher = threading.Thread(
        target=_executor_status_watcher,
        args=(stop_event, time.time()),
        daemon=True,
    )
    watcher.start()

    try:
        # Executor is persistent. The lock is held by generate(), so no
        # second generation can enter this section concurrently.
        log("[execute] executor.execute() START")
        _COMFY_EXECUTOR.execute(
            workflow,
            prompt_id,
            extra_data={},
            execute_outputs=[save_id],
        )
        log("[execute] executor.execute() RETURNED")
    except BaseException as exc:
        log_exception("executor.execute", exc)
        raise
    finally:
        stop_event.set()
        watcher.join(timeout=1)

    if not _COMFY_EXECUTOR.success:
        messages = getattr(_COMFY_EXECUTOR, "status_messages", None) or []
        message = messages[-1] if messages else "ComfyUI execution failed"
        log("[execute] executor reports failure: %s", message)
        raise RuntimeError(str(message))

    paths: list[Path] = []

    history = getattr(_COMFY_EXECUTOR, "history_result", {}) or {}
    for output in history.get("outputs", {}).values():
        for items in output.values():
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict) or not item.get("filename"):
                    continue

                item_type = item.get("type", "output")
                base = OUTPUT if item_type == "output" else COMFY / item_type
                candidate = base / item.get("subfolder", "") / item["filename"]
                if candidate.exists():
                    paths.append(candidate)

    if not paths:
        paths = sorted(
            [Path(x) for x in glob.glob(str(OUTPUT / "**" / "*.png"), recursive=True)],
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )

    if not paths:
        raise RuntimeError("ComfyUI finished without an output image")

    log("[execute] found %d output image(s)", len(paths))
    return [str(x) for x in paths]

# ============================================================================
# RUNTIME
# ============================================================================

def _prepare_runtime(base_model: str) -> str:
    log("[runtime] ensuring ComfyUI")
    _ensure_comfy()
    _scan_local_models()
    _validate_required_assets()

    if not LOCAL_BASE_MODELS:
        raise RuntimeError(f"No diffusion models found in {DIFFUSION_DIR}")

    resolved = _validate_model_name(base_model)
    _init_comfy_nodes()
    return resolved

def get_gpu_duration(*args: Any, **kwargs: Any) -> int:
    steps = kwargs.get("steps", args[11] if len(args) > 11 else DEFAULT_STEPS)
    width = kwargs.get("width", args[5] if len(args) > 5 else DEFAULT_WIDTH)
    height = kwargs.get("height", args[6] if len(args) > 6 else DEFAULT_HEIGHT)
    budget = kwargs.get("gen_budget", args[17] if len(args) > 17 else 0)

    if budget and int(budget) > 0:
        return max(MIN_GPU_SECONDS, min(MAX_GPU_SECONDS, int(budget)))

    weights = kwargs.get("lora_weights", args[19] if len(args) > 19 else {}) or {}
    count = sum(
        1 for v in weights.values()
        if v and abs(float(v)) > 1e-6
    )

    estimate = int(
        35 + (int(width) * int(height) / 1_000_000)
        * int(steps) * 3.0 * (1 + 0.05 * count)
    )
    return max(MIN_GPU_SECONDS, min(MAX_GPU_SECONDS, estimate))

# ============================================================================
# REQUEST VALIDATION
# ============================================================================

def _validate_request(mode: str, prompt: str, edit_prompt: str, primary: str | None) -> None:
    if mode not in {"text2image", "edit"}:
        raise ValueError("unsupported generation mode")

    if mode == "text2image" and not (prompt or "").strip():
        raise ValueError("enter a prompt")

    if mode == "edit":
        if not primary:
            raise ValueError("upload a primary image for edit mode")
        if not (edit_prompt or prompt or "").strip():
            raise ValueError("enter an edit instruction")

# ============================================================================
# GPU / PROFILE HELPERS
# ============================================================================

def _profile_from_values(
    mode, prompt, edit_prompt, width, height, target_mp, grounding,
    ref_boost, ref_boost_a, steps, cfg, sampler, scheduler, seed,
    randomize, budget, base_model, catalog_loras=None,
):
    return build_settings(
        mode=mode, prompt=prompt, edit_prompt=edit_prompt,
        width=int(width), height=int(height),
        target_megapixels=float(target_mp),
        grounding_px=int(grounding),
        ref_boost=float(ref_boost), ref_boost_a=float(ref_boost_a),
        steps=int(steps), cfg=float(cfg),
        sampler_name=sampler, scheduler=scheduler,
        seed=int(seed), randomize_seed=bool(randomize),
        gen_budget=float(budget), effective_seed=int(seed),
        base_model=base_model, custom_base_model=None,
        catalog_loras=catalog_loras, custom_loras=[],
    )

# ============================================================================
# GENERATION IMPLEMENTATION
# ============================================================================

# Standalone-generation settings are overwritten by generate_once().
MODE = "text2image"
PROMPT = "A cinematic portrait in soft natural light"
EDIT_PROMPT = ""
PRIMARY_IMAGE = None
SECOND_IMAGE = None
WIDTH = 1024
HEIGHT = 1024
TARGET_MEGAPIXELS = 1.4
GROUNDING_PX = 768
REF_BOOST = 1.0
REF_BOOST_A = 1.0
STEPS = 8
CFG = 1.0
SAMPLER = "euler"
SCHEDULER = "beta"
SEED = 2
RANDOMIZE_SEED = False
GEN_BUDGET = 0
BASE_MODEL = ""
LORA_WEIGHTS: dict[str, float] = {}


def run_cell_generation() -> tuple[list[str], int]:
    total_start = time.time()
    staged: list[Path] = []

    log("=" * 80)
    log("[generate] ===== START =====")
    log("[generate] mode=%s model=%s", MODE, BASE_MODEL)
    log("[config] size=%sx%s steps=%s cfg=%s", WIDTH, HEIGHT, STEPS, CFG)
    log("[config] sampler=%s scheduler=%s seed=%s", SAMPLER, SCHEDULER, SEED)
    log("[config] LoRAs=%s", LORA_WEIGHTS)
    _memory_snapshot("generation-start")

    if not _GENERATION_LOCK.acquire(timeout=600):
        raise RuntimeError("Could not acquire generation lock within 600 seconds")

    try:
        log("[generate] generation lock acquired")

        _validate_request(MODE, PROMPT, EDIT_PROMPT, PRIMARY_IMAGE)
        log("[generate] stage 1: validate request")

        effective_edit_prompt = (EDIT_PROMPT or PROMPT or "").strip()

        if SAMPLER not in SAMPLERS:
            raise ValueError(f"unsupported sampler: {SAMPLER}")
        if SCHEDULER not in SCHEDULERS:
            raise ValueError(f"unsupported scheduler: {SCHEDULER}")

        # IMPORTANT: this only checks/uses the already initialized runtime.
        # _init_comfy_nodes() is idempotent and will not create another executor.
        log("[generate] stage 2: validate runtime")
        _ensure_comfy()
        _scan_local_models()
        _validate_required_assets()

        resolved_base_model = _validate_model_name(BASE_MODEL)
        _init_comfy_nodes()

        enabled_loras: list[tuple[str, float]] = []
        for filename, weight in (LORA_WEIGHTS or {}).items():
            if filename not in LOCAL_LORAS:
                raise ValueError(f"local LoRA is not available: {filename}")

            numeric = float(weight)
            if numeric < -3.0 or numeric > 3.0:
                raise ValueError(f"LoRA weight out of range: {filename}")

            if abs(numeric) > 1e-6:
                enabled_loras.append((_validate_lora_name(filename), numeric))

        effective_seed = (
            random.randint(0, 2**32 - 1)
            if RANDOMIZE_SEED or int(SEED) < 0
            else int(SEED)
        )

        if MODE == "text2image":
            width = max(512, min(MAX_WIDTH, int(WIDTH) // 64 * 64))
            height = max(512, min(MAX_HEIGHT, int(HEIGHT) // 64 * 64))
            log("[generate] stage 3: build text2image workflow")
            workflow = _t2i_workflow(resolved_base_model)
        else:
            log("[generate] stage 3: prepare edit image")
            primary_name, width, height = _prepare_edit_image(
                PRIMARY_IMAGE,
                TARGET_MEGAPIXELS,
            )
            staged.append(INPUT / primary_name)

            second_name = (
                _stage_image(SECOND_IMAGE, "reference")
                if SECOND_IMAGE
                else None
            )
            if second_name:
                staged.append(INPUT / second_name)

            log("[generate] stage 4: build edit workflow")
            workflow = _edit_workflow(bool(second_name), resolved_base_model)

        if MODE == "text2image":
            log("[generate] stage 4: inject parameters and LoRAs")
            _inject_t2i(
                workflow,
                prompt=PROMPT,
                width=width,
                height=height,
                steps=int(STEPS),
                cfg=float(CFG),
                sampler=SAMPLER,
                scheduler=SCHEDULER,
                seed=effective_seed,
                enabled_loras=enabled_loras,
            )
        else:
            log("[generate] stage 5: inject parameters and LoRAs")
            _inject_edit(
                workflow,
                primary_name=primary_name,
                second_name=second_name,
                width=width,
                height=height,
                edit_prompt=effective_edit_prompt,
                grounding_px=int(GROUNDING_PX),
                ref_boost=float(REF_BOOST),
                ref_boost_a=float(REF_BOOST_A),
                steps=int(STEPS),
                cfg=float(CFG),
                sampler=SAMPLER,
                scheduler=SCHEDULER,
                seed=effective_seed,
                enabled_loras=enabled_loras,
            )

        active_loras = [
            {"hf_filename": filename, "weight": float(weight)}
            for filename, weight in enabled_loras
        ]

        log("[generate] stage 6: build metadata")
        settings = build_settings(
            mode=MODE,
            prompt=PROMPT,
            edit_prompt=effective_edit_prompt,
            width=width,
            height=height,
            target_megapixels=float(TARGET_MEGAPIXELS),
            grounding_px=int(GROUNDING_PX),
            ref_boost=float(REF_BOOST),
            ref_boost_a=float(REF_BOOST_A),
            steps=int(STEPS),
            cfg=float(CFG),
            sampler_name=SAMPLER,
            scheduler=SCHEDULER,
            seed=int(SEED),
            randomize_seed=bool(RANDOMIZE_SEED),
            gen_budget=float(GEN_BUDGET),
            effective_seed=effective_seed,
            base_model=BASE_MODEL,
            custom_base_model=None,
            catalog_loras=active_loras,
            custom_loras=[],
        )

        log("[generate] stage 7: execute ComfyUI workflow")
        result_paths = _execute_workflow(workflow)

        log("[generate] stage 8: write metadata-preserving output")
        destination_dir = Path(tempfile.mkdtemp(prefix="krea2_outputs_"))
        output_paths: list[str] = []

        for index, source in enumerate(result_paths):
            destination = destination_dir / f"output_{index}.png"
            write_png_metadata(source, destination, settings)
            output_paths.append(str(destination))
            log("[output] %s", destination)

        if MEGA_ACCOUNT is None:
            raise RuntimeError("MEGA account is not initialized")

        log("[generate] stage 9: upload to MEGA")
        for output_path in output_paths:
            _upload_to_mega(output_path, MEGA_ACCOUNT)

        elapsed = time.time() - total_start
        log("[mega] uploaded %d image(s)", len(output_paths))
        log("[generate] ===== SUCCESS in %.1fs =====", elapsed)
        log("[generate] used seed=%s", effective_seed)
        _memory_snapshot("generation-success")

        return output_paths, effective_seed

    except BaseException as exc:
        log_exception("generate", exc)
        _memory_snapshot("generation-error")
        raise

    finally:
        log("[generate] cleanup: %d staged file(s)", len(staged))

        for path in staged:
            try:
                path.unlink(missing_ok=True)
            except Exception as exc:
                log(
                    "[cleanup] failed removing %s: %s: %s",
                    path,
                    type(exc).__name__,
                    exc,
                )

        # This cleans temporary CUDA allocations but deliberately does NOT
        # recreate or destroy the persistent ComfyUI runtime.
        _cleanup_memory("generation-finally")

        try:
            _GENERATION_LOCK.release()
            log("[generate] generation lock released")
        except RuntimeError:
            pass

        log("[generate] cleanup complete")
        log("=" * 80)


# ============================================================================
# NOTEBOOK / PERSISTENT RUNTIME API
# ============================================================================

# This file is intentionally importable from Google Colab.
#
# Recommended notebook flow:
#   Cell 1: scan model/LoRA files
#   Cell 2: set generation variables
#   Cell 3: import app_v2 and preload()
#   Cell 4: generate_once(...), then display the returned image
#
# DO NOT run `!python app_v2.py` for every image.  That starts a new Python
# process and loses the persistent ComfyUI runtime.

DEFAULT_CONFIG = {
    "MODE": "text2image",
    "PROMPT": "A cinematic portrait in soft natural light",
    "EDIT_PROMPT": "",
    "PRIMARY_IMAGE": None,
    "SECOND_IMAGE": None,
    "WIDTH": 1024,
    "HEIGHT": 1024,
    "TARGET_MEGAPIXELS": 1.4,
    "GROUNDING_PX": 768,
    "REF_BOOST": 1.0,
    "REF_BOOST_A": 1.0,
    "STEPS": 8,
    "CFG": 1.0,
    "SAMPLER": "euler",
    "SCHEDULER": "beta",
    "SEED": 2,
    "RANDOMIZE_SEED": False,
    "GEN_BUDGET": 0,
    "BASE_MODEL": "",
    "LORA_WEIGHTS": {},
}


def _display_generated_image(path: str) -> None:
    """Display an image when app_v2 is imported in a notebook."""
    try:
        from IPython.display import display
        with Image.open(path) as image:
            display(image.copy())
    except Exception as exc:
        log_exception("image-display", exc)
        print(f"Generated image: {path}", flush=True)


def preload() -> dict[str, Any]:
    """Initialize ComfyUI/MEGA once and keep the runtime alive.

    Call this ONCE in a Colab cell. Subsequent calls are harmless.
    """
    global MEGA_ACCOUNT

    with _COMFY_RUNTIME_LOCK:
        if MEGA_ACCOUNT is None:
            log("[preload] logging into MEGA")
            MEGA_ACCOUNT = _mega_login()
        else:
            log("[preload] MEGA already initialized")

        log("[preload] ensuring ComfyUI")
        _ensure_comfy()
        _scan_local_models()
        _validate_required_assets()

        if not LOCAL_BASE_MODELS:
            raise RuntimeError(
                f"No diffusion models found in {DIFFUSION_DIR}"
            )

        log("[preload] initializing persistent ComfyUI runtime")
        _init_comfy_nodes()

        log("=" * 80)
        log("[preload] READY — runtime will stay loaded in this notebook kernel")
        log("[preload] base models: %d", len(LOCAL_BASE_MODELS))
        log("[preload] LoRAs: %d", len(LOCAL_LORAS))
        log("=" * 80)
        _memory_snapshot("preload-ready")

        return {
            "comfy": str(COMFY),
            "models": list(LOCAL_BASE_MODELS),
            "loras": list(LOCAL_LORAS),
            "ready": bool(_NODES_READY and _COMFY_EXECUTOR is not None),
        }


def _normalise_config(config: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(DEFAULT_CONFIG)
    if config:
        merged.update(config)

    if not merged["BASE_MODEL"]:
        if not LOCAL_BASE_MODELS:
            raise RuntimeError("No base models are available")
        merged["BASE_MODEL"] = LOCAL_BASE_MODELS[0]

    return merged


def generate_once(config: dict[str, Any] | None = None) -> tuple[list[str], int]:
    """Generate one image using the already-preloaded runtime.

    This is the function your Colab generation cells should call.
    It does NOT recreate PromptServer, PromptExecutor, or ComfyUI nodes.
    """
    cfg = _normalise_config(config)

    # Keep the existing generation implementation but feed it the current
    # notebook settings.  The runtime objects themselves remain persistent.
    global MODE, PROMPT, EDIT_PROMPT, PRIMARY_IMAGE, SECOND_IMAGE
    global WIDTH, HEIGHT, TARGET_MEGAPIXELS, GROUNDING_PX
    global REF_BOOST, REF_BOOST_A, STEPS, CFG, SAMPLER, SCHEDULER
    global SEED, RANDOMIZE_SEED, GEN_BUDGET, BASE_MODEL, LORA_WEIGHTS

    MODE = cfg["MODE"]
    PROMPT = cfg["PROMPT"]
    EDIT_PROMPT = cfg["EDIT_PROMPT"]
    PRIMARY_IMAGE = cfg["PRIMARY_IMAGE"]
    SECOND_IMAGE = cfg["SECOND_IMAGE"]
    WIDTH = int(cfg["WIDTH"])
    HEIGHT = int(cfg["HEIGHT"])
    TARGET_MEGAPIXELS = float(cfg["TARGET_MEGAPIXELS"])
    GROUNDING_PX = int(cfg["GROUNDING_PX"])
    REF_BOOST = float(cfg["REF_BOOST"])
    REF_BOOST_A = float(cfg["REF_BOOST_A"])
    STEPS = int(cfg["STEPS"])
    globals()["CFG"] = float(cfg["CFG"])
    SAMPLER = cfg["SAMPLER"]
    SCHEDULER = cfg["SCHEDULER"]
    SEED = int(cfg["SEED"])
    RANDOMIZE_SEED = bool(cfg["RANDOMIZE_SEED"])
    GEN_BUDGET = float(cfg["GEN_BUDGET"])
    BASE_MODEL = cfg["BASE_MODEL"]
    LORA_WEIGHTS = dict(cfg.get("LORA_WEIGHTS") or {})

    if not _NODES_READY or _COMFY_EXECUTOR is None:
        log("[generate] runtime not ready; preloading now")
        preload()

    return run_cell_generation()


def runtime_status() -> dict[str, Any]:
    """Return a small status dictionary useful from Colab."""
    return {
        "comfy_ready": _COMFY_READY,
        "nodes_ready": _NODES_READY,
        "executor_ready": _COMFY_EXECUTOR is not None,
        "mega_ready": MEGA_ACCOUNT is not None,
        "base_models": list(LOCAL_BASE_MODELS),
        "loras": list(LOCAL_LORAS),
    }


# Backwards-compatible standalone mode. This is useful for testing, but it is
# NOT the recommended way to generate multiple images from Colab.
if __name__ == "__main__":
    try:
        log("=" * 80)
        log("Krea 2 Turbo - standalone mode")
        log("[warning] For repeated Colab generation, import app_v2 and use preload().")
        log("=" * 80)
        preload()
        outputs, used_seed = generate_once()
        log("[main] generation successful; seed=%s", used_seed)
        for output in outputs:
            log("[main] output=%s", output)
            _display_generated_image(output)
    except BaseException as exc:
        log_exception("main", exc)
        _cleanup_memory("application-failure")
        raise
