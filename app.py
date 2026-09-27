from __future__ import annotations

import asyncio
import glob
import json
import os
import pathlib
import random
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
import uuid
from typing import Any
import time
import threading
from datetime import datetime

import gradio as gr
from huggingface_hub import hf_hub_download
from PIL import Image
from pathlib import Path


# ============================================================================
# MEGA UPLOAD
# ============================================================================

_MEGA_UPLOAD_LOCK = threading.Lock()


def _mega_login():
    """Log in to MEGA once at application startup and fail early if invalid."""
    email = os.environ.get("MEGA_EMAIL")
    password = os.environ.get("MEGA_PASSWORD")

    if not email or not password:
        raise RuntimeError(
            "Missing MEGA_EMAIL or MEGA_PASSWORD in Spaces secrets. "
            "Set MEGA_EMAIL and MEGA_PASSWORD before starting the app."
        )

    try:
        from mega import Mega
    except ImportError as exc:
        raise RuntimeError(
            "The MEGA package is missing. Add mega.py-v2 to requirements.txt."
        ) from exc

    try:
        account = Mega().login(email, password)
        # Force a real authenticated request so invalid credentials are
        # detected before ComfyUI setup and before the Gradio UI is created.
        account.get_files()
    except Exception as exc:
        raise RuntimeError(
            "MEGA login verification failed. Check MEGA_EMAIL, "
            "MEGA_PASSWORD, and the installed MEGA package."
        ) from exc

    print("[mega] login verification successful", flush=True)
    return account


def _mega_remote_filenames(account) -> set[str]:
    """Return all filenames currently visible in the MEGA account."""
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


def _unique_mega_filename(existing_names: set[str], extension: str = ".png") -> str:
    """Create the requested timestamp filename and avoid collisions."""
    timestamp = datetime.now().strftime("Image %b %d, %Y, %I_%M_%S %p")
    base = f"{timestamp}{extension}"

    if base not in existing_names:
        return base

    counter = 1
    while True:
        candidate = f"{timestamp}_{counter:03d}{extension}"
        if candidate not in existing_names:
            return candidate
        counter += 1


def _upload_to_mega(file_path: str, account) -> str:
    """Upload a generated image to MEGA with a collision-safe filename."""
    path = pathlib.Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Generated file does not exist: {file_path}")

    with _MEGA_UPLOAD_LOCK:
        existing_names = _mega_remote_filenames(account)
        filename = _unique_mega_filename(
            existing_names,
            extension=path.suffix or ".png",
        )

        # mega.py supports dest_filename for naming the uploaded remote file.
        uploaded = account.upload(
            str(path),
            dest=None,
            dest_filename=filename,
        )

        print(f"[mega] uploaded: {filename}", flush=True)
        return str(uploaded)


# Verify MEGA before any ComfyUI setup, model scanning, or UI creation.
MEGA_ACCOUNT = _mega_login()


# ============================================================================
# SPACES COMPATIBILITY
# ============================================================================

try:
    import spaces
except ImportError:

    class _SpacesFallback:

        @staticmethod
        def GPU(**_kwargs):

            def decorate(function):
                return function

            return decorate

    spaces = _SpacesFallback()


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

ROOT = pathlib.Path(__file__).resolve().parent


def _detect_comfy_root() -> pathlib.Path:

    # Case 1:
    # app.py is directly inside ComfyUI.
    if (
        (ROOT / "main.py").is_file()
        and (ROOT / "models").is_dir()
    ):
        return ROOT

    # Case 2:
    # ComfyUI is a child of the application directory.
    candidate = ROOT / "ComfyUI"

    if (
        (candidate / "main.py").is_file()
        and (candidate / "models").is_dir()
    ):
        return candidate

    # Case 3:
    # /content/ComfyUI is the standard Colab location.
    candidate = pathlib.Path("/content/ComfyUI")

    if (
        (candidate / "main.py").is_file()
        and (candidate / "models").is_dir()
    ):
        return candidate

    # Last resort.
    return ROOT / "ComfyUI"


COMFY = _detect_comfy_root()

MODELS = Path("/mnt/krea2-models")

import folder_paths

folder_paths.add_model_folder_path(
    "diffusion_models",
    str(MODELS / "diffusion_models")
)

folder_paths.add_model_folder_path(
    "loras",
    str(MODELS / "loras")
)

folder_paths.add_model_folder_path(
    "vae",
    str(MODELS / "vae")
)

folder_paths.add_model_folder_path(
    "text_encoders",
    str(MODELS / "text_encoders")
)

INPUT = COMFY / "input"
OUTPUT = COMFY / "output"
CUSTOM_NODES = COMFY / "custom_nodes"


print(
    "[paths] ROOT:",
    ROOT,
    flush=True,
)

print(
    "[paths] COMFY:",
    COMFY,
    flush=True,
)

print(
    "[paths] MODELS:",
    MODELS,
    flush=True,
)


# ============================================================================
# WORKFLOW FILES
# ============================================================================

T2I_SOURCE = ROOT / "lustifyWorkflowsKrea2_krea2.json"
EDIT_SOURCE = ROOT / "lustifyWorkflowsKrea2_krea2Edit.json"


# ============================================================================
# KREA EDIT NODE
# ============================================================================

KREA_EDIT_NODES = (
    "https://github.com/lbouaraba/comfyui-krea2edit.git"
)


# ============================================================================
# IDENTITY ADAPTER
# ============================================================================

IDENTITY_REPO = "conradlocke/krea2-identity-edit"

IDENTITY_FILE = "krea2_identity_edit_v1_2.safetensors"

IDENTITY_LORA_DIR = (
    MODELS / "loras" / "krea"
)

IDENTITY_LORA_PATH = (
    IDENTITY_LORA_DIR / IDENTITY_FILE
)

IDENTITY_COMFY_NAME = pathlib.PurePosixPath(
    "krea",
    IDENTITY_FILE,
).as_posix()


# ============================================================================
# LOCAL MODEL DIRECTORIES
# ============================================================================

TEXT_ENCODER_DIR = MODELS / "text_encoders"

VAE_DIR = MODELS / "vae"

DIFFUSION_DIR = MODELS / "diffusion_models"

LORA_ROOT = MODELS / "loras"


# ============================================================================
# REQUIRED KREA FILES
# ============================================================================

TEXT_ENCODER_FILE = (
    "qwen3vl_4b_fp8_scaled.safetensors"
)

VAE_FILE = (
    "qwen_image_vae.safetensors"
)


# ============================================================================
# MODEL CATALOGS
# ============================================================================

LOCAL_BASE_MODELS: list[str] = []

LOCAL_LORAS: list[str] = []


# ============================================================================
# SAMPLERS
# ============================================================================

SAMPLERS = [
    "euler",
    "euler_ancestral",
    "euler_a",
    "dpmpp_2m",
    "dpmpp_2m_sde",
    "dpmpp_sde",
    "heun",
    "lms",
]


SCHEDULERS = [
    "beta",
    "normal",
    "karras",
    "exponential",
    "sgm_uniform",
    "simple",
]


# ============================================================================
# DEFAULTS
# ============================================================================

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


MIN_GPU_SECONDS = int(
    os.environ.get(
        "MIN_GPU_SECONDS",
        "45",
    )
)


MAX_GPU_SECONDS = int(
    os.environ.get(
        "MAX_GPU_SECONDS",
        "300",
    )
)


# ============================================================================
# RUNTIME STATE
# ============================================================================

_comfy_ready = False

_nodes_ready = False

_workflow_cache: dict[str, dict[str, Any]] = {}


# ============================================================================
# COMMAND HELPERS
# ============================================================================

def _run(
    command: list[str],
    cwd: pathlib.Path | None = None,
    check: bool = True,
) -> None:

    print(
        "[setup]",
        " ".join(command),
        flush=True,
    )

    subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        check=check,
    )


def _pip_install(
    arguments: list[str],
) -> None:

    _run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            *arguments,
        ],
        check=False,
    )


def _install_filtered_requirements(
    path: pathlib.Path,
) -> None:

    if not path.exists():
        return

    blocked = {
        "torch",
        "torchvision",
        "torchaudio",
        "transformers",
        "huggingface-hub",
        "accelerate",
    }

    requirements: list[str] = []

    for raw in path.read_text(
        encoding="utf-8",
        errors="ignore",
    ).splitlines():

        item = raw.strip()

        if not item:
            continue

        if item.startswith("#"):
            continue

        package = re.split(
            r"[<>=!~;\[\s]",
            item.lower().replace("_", "-"),
            maxsplit=1,
        )[0]

        if package not in blocked:
            requirements.append(item)

    if requirements:
        _pip_install(requirements)


def _ensure_repo(
    path: pathlib.Path,
    url: str,
) -> None:

    if path.exists():
        return

    _run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            url,
            str(path),
        ]
    )


# ============================================================================
# COMFY UTILS COMPATIBILITY
# ============================================================================

def _restore_utils_namespace() -> None:

    source = COMFY / "utils"

    target = COMFY / "utilities"

    if not source.exists() and target.exists():

        target.rename(source)

    if not source.exists():
        return

    for path in COMFY.rglob("*.py"):

        if "__pycache__" in path.parts:
            continue

        try:

            text = path.read_text(
                encoding="utf-8"
            )

        except UnicodeDecodeError:

            continue

        updated = re.sub(
            r"\bfrom utilities\b",
            "from utils",
            text,
        )

        updated = re.sub(
            r"\bimport utilities\b",
            "import utils",
            updated,
        )

        if updated != text:

            path.write_text(
                updated,
                encoding="utf-8",
            )


# ============================================================================
# DIRECTORY SETUP
# ============================================================================

def _ensure_model_directories() -> None:

    for folder in [
        "diffusion_models",
        "text_encoders",
        "vae",
        "loras",
        "loras/krea",
    ]:

        (
            MODELS / folder
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

    INPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================================
# IDENTITY MODEL
# ============================================================================

def _download_identity_model() -> None:

    IDENTITY_LORA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if IDENTITY_LORA_PATH.exists():

        print(
            "[identity] already installed:",
            IDENTITY_LORA_PATH,
            flush=True,
        )

        return

    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get(
            "HUGGINGFACE_HUB_TOKEN"
        )
    )

    print(
        "[identity] downloading:",
        IDENTITY_FILE,
        flush=True,
    )

    downloaded = pathlib.Path(
        hf_hub_download(
            repo_id=IDENTITY_REPO,
            filename=IDENTITY_FILE,
            local_dir=str(
                IDENTITY_LORA_DIR
            ),
            token=token,
        )
    )

    if (
        downloaded.resolve()
        != IDENTITY_LORA_PATH.resolve()
    ):

        shutil.move(
            str(downloaded),
            str(IDENTITY_LORA_PATH),
        )

    print(
        "[identity] ready:",
        IDENTITY_LORA_PATH,
        flush=True,
    )


# ============================================================================
# LOCAL MODEL SCANNER
# ============================================================================

def _scan_local_models() -> None:

    global LOCAL_BASE_MODELS
    global LOCAL_LORAS

    _ensure_model_directories()

    extensions = {
        ".safetensors",
        ".ckpt",
        ".pt",
        ".bin",
    }


    # ------------------------------------------------------------------------
    # DIFFUSION MODELS
    # ------------------------------------------------------------------------

    models: list[str] = []

    if DIFFUSION_DIR.exists():

        for path in DIFFUSION_DIR.rglob("*"):

            if not path.is_file():
                continue

            if path.suffix.lower() not in extensions:
                continue

            relative = path.relative_to(
                DIFFUSION_DIR
            )

            models.append(
                pathlib.PurePosixPath(
                    *relative.parts
                ).as_posix()
            )

    LOCAL_BASE_MODELS = sorted(
        models,
        key=str.lower,
    )


    # ------------------------------------------------------------------------
    # LORAS
    # ------------------------------------------------------------------------

    loras: list[str] = []

    if LORA_ROOT.exists():

        for path in LORA_ROOT.rglob("*"):

            if not path.is_file():
                continue

            if path.suffix.lower() not in extensions:
                continue

            try:

                if (
                    path.resolve()
                    == IDENTITY_LORA_PATH.resolve()
                ):

                    continue

            except OSError:

                pass

            relative = path.relative_to(
                LORA_ROOT
            )

            loras.append(
                pathlib.PurePosixPath(
                    *relative.parts
                ).as_posix()
            )

    LOCAL_LORAS = sorted(
        loras,
        key=str.lower,
    )


    # ------------------------------------------------------------------------
    # LOG
    # ------------------------------------------------------------------------

    print(
        f"[models] found "
        f"{len(LOCAL_BASE_MODELS)} "
        f"local diffusion model(s)",
        flush=True,
    )

    for model in LOCAL_BASE_MODELS:

        print(
            "[models]  ",
            model,
            flush=True,
        )


    print(
        f"[loras] found "
        f"{len(LOCAL_LORAS)} "
        f"local LoRA(s)",
        flush=True,
    )

    for lora in LOCAL_LORAS:

        print(
            "[loras]  ",
            lora,
            flush=True,
        )


# ============================================================================
# REQUIRED ASSETS
# ============================================================================

def _validate_required_assets() -> None:

    missing: list[str] = []


    text_encoder = (
        TEXT_ENCODER_DIR
        / TEXT_ENCODER_FILE
    )

    if not text_encoder.is_file():

        missing.append(
            f"text encoder: {text_encoder}"
        )


    vae = (
        VAE_DIR
        / VAE_FILE
    )

    if not vae.is_file():

        missing.append(
            f"VAE: {vae}"
        )


    if not IDENTITY_LORA_PATH.is_file():

        missing.append(
            f"identity adapter: "
            f"{IDENTITY_LORA_PATH}"
        )


    if missing:

        raise RuntimeError(
            "Missing required local model files:\n"
            + "\n".join(
                f"  - {item}"
                for item in missing
            )
        )


# ============================================================================
# COMFY SETUP
# ============================================================================

def _ensure_comfy() -> None:

    global _comfy_ready

    if _comfy_ready:
        return


    print(
        "[comfy] using:",
        COMFY,
        flush=True,
    )


    if not (
        COMFY / "main.py"
    ).exists():

        raise RuntimeError(
            "ComfyUI was not found at:\n"
            f"{COMFY}"
        )


    # ------------------------------------------------------------------------
    # Requirements
    # ------------------------------------------------------------------------

    _install_filtered_requirements(
        COMFY / "requirements.txt"
    )


    # ------------------------------------------------------------------------
    # Custom nodes
    # ------------------------------------------------------------------------

    CUSTOM_NODES.mkdir(
        parents=True,
        exist_ok=True,
    )

    _ensure_repo(
        CUSTOM_NODES / "comfyui-krea2edit",
        KREA_EDIT_NODES,
    )


    # ------------------------------------------------------------------------
    # Compatibility
    # ------------------------------------------------------------------------

    _restore_utils_namespace()


    # ------------------------------------------------------------------------
    # Directories
    # ------------------------------------------------------------------------

    _ensure_model_directories()


    # ------------------------------------------------------------------------
    # Identity adapter only
    # ------------------------------------------------------------------------

    _download_identity_model()


    # ------------------------------------------------------------------------
    # Scan
    # ------------------------------------------------------------------------

    _scan_local_models()


    _comfy_ready = True


# ============================================================================
# NODE INITIALIZATION
# ============================================================================

def _init_comfy_nodes() -> None:

    global _nodes_ready

    if _nodes_ready:
        return


    comfy_path = str(COMFY)


    # Make sure ComfyUI is first.
    sys.path = [
        item
        for item in sys.path
        if item != comfy_path
    ]

    sys.path.insert(
        0,
        comfy_path,
    )


    # Remove stale utils modules.
    for name in list(sys.modules):

        if (
            name == "utils"
            or name.startswith("utils.")
        ):

            del sys.modules[name]


    os.chdir(COMFY)


    import execution
    import nodes
    import server

    from app.assets.manager import default_asset_manager
    loop = asyncio.new_event_loop()

    asyncio.set_event_loop(loop)


    server_instance = server.PromptServer(
        loop,
        default_asset_manager()
    )


    execution.PromptQueue(
        server_instance
    )


    loop.run_until_complete(
        nodes.init_extra_nodes()
    )


    _nodes_ready = True


# ============================================================================
# MODEL VALIDATION
# ============================================================================

def _validate_model_name(
    model_name: str,
) -> str:

    normalized = str(
        model_name
    ).replace(
        "\\",
        "/",
    )

    path = pathlib.PurePosixPath(
        normalized
    )


    if (
        not normalized
        or path.is_absolute()
        or any(
            part in {
                "",
                ".",
                "..",
            }
            for part in path.parts
        )
    ):

        raise ValueError(
            "invalid diffusion model path"
        )


    candidate = (
        DIFFUSION_DIR
        / pathlib.Path(
            *path.parts
        )
    )


    try:

        candidate.resolve().relative_to(
            DIFFUSION_DIR.resolve()
        )

    except ValueError as exc:

        raise ValueError(
            "invalid diffusion model path"
        ) from exc


    if not candidate.is_file():

        raise ValueError(
            "diffusion model is not installed: "
            + normalized
        )


    return path.as_posix()


# ============================================================================
# LORA VALIDATION
# ============================================================================

def _validate_lora_name(
    lora_name: str,
) -> str:

    normalized = str(
        lora_name
    ).replace(
        "\\",
        "/",
    )

    path = pathlib.PurePosixPath(
        normalized
    )


    if (
        not normalized
        or path.is_absolute()
        or any(
            part in {
                "",
                ".",
                "..",
            }
            for part in path.parts
        )
    ):

        raise ValueError(
            "invalid LoRA path"
        )


    candidate = (
        LORA_ROOT
        / pathlib.Path(
            *path.parts
        )
    )


    try:

        candidate.resolve().relative_to(
            LORA_ROOT.resolve()
        )

    except ValueError as exc:

        raise ValueError(
            "invalid LoRA path"
        ) from exc


    if not candidate.is_file():

        raise ValueError(
            "LoRA is not installed: "
            + normalized
        )


    if (
        candidate.resolve()
        == IDENTITY_LORA_PATH.resolve()
    ):

        raise ValueError(
            "identity adapter cannot be "
            "selected as a user LoRA"
        )


    return path.as_posix()


# ============================================================================
# WORKFLOW HELPERS
# ============================================================================

def _read_source_workflow(
    path: pathlib.Path,
) -> dict[str, Any]:

    if not path.exists():

        raise FileNotFoundError(
            f"workflow file is missing: "
            f"{path}"
        )


    data = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


    if not data.get("nodes"):

        raise ValueError(
            f"workflow file has no nodes: "
            f"{path.name}"
        )


    return data


def _ref(
    node: str,
    output: int = 0,
) -> list[Any]:

    return [
        node,
        output,
    ]


# ============================================================================
# T2I WORKFLOW
# ============================================================================

def _t2i_workflow(
    base_model: str,
) -> dict[str, Any]:

    base_model = _validate_model_name(
        base_model
    )


    cache_key = (
        f"text2image:{base_model}"
    )


    if cache_key in _workflow_cache:

        return json.loads(
            json.dumps(
                _workflow_cache[
                    cache_key
                ]
            )
        )


    _read_source_workflow(
        T2I_SOURCE
    )


    workflow = {

        "1": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": base_model,
                "weight_dtype": "default",
            },
        },

        "2": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": TEXT_ENCODER_FILE,
                "type": "krea2",
                "device": "default",
            },
        },

        "3": {
            "class_type": "VAELoader",
            "inputs": {
                "vae_name": VAE_FILE,
            },
        },

        "4": {
            "class_type": "ModelSamplingAuraFlow",
            "inputs": {
                "model": _ref("1"),
                "shift": 4.0,
            },
        },

        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": _ref("2"),
                "text": "",
            },
        },

        "6": {
            "class_type": "ConditioningZeroOut",
            "inputs": {
                "conditioning": _ref("5"),
            },
        },

        "7": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": DEFAULT_WIDTH,
                "height": DEFAULT_HEIGHT,
                "batch_size": 1,
            },
        },

        "8": {
            "class_type": "KSampler",
            "inputs": {
                "model": _ref("4"),
                "positive": _ref("5"),
                "negative": _ref("6"),
                "latent_image": _ref("7"),
                "seed": DEFAULT_SEED,
                "steps": DEFAULT_STEPS,
                "cfg": DEFAULT_CFG,
                "sampler_name": DEFAULT_SAMPLER,
                "scheduler": DEFAULT_SCHEDULER,
                "denoise": 1.0,
            },
        },

        "9": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": _ref("8"),
                "vae": _ref("3"),
            },
        },

        "10": {
            "class_type": "SaveImage",
            "inputs": {
                "images": _ref("9"),
                "filename_prefix": "krea2_turbo",
            },
        },
    }


    _workflow_cache[
        cache_key
    ] = workflow


    return json.loads(
        json.dumps(workflow)
    )


# ============================================================================
# EDIT WORKFLOW
# ============================================================================

def _edit_workflow(
    has_second_reference: bool,
    base_model: str,
) -> dict[str, Any]:

    base_model = _validate_model_name(
        base_model
    )


    _read_source_workflow(
        EDIT_SOURCE
    )


    workflow: dict[str, Any] = {

        "1": {
            "class_type": "LoadImage",
            "inputs": {
                "image": "",
            },
        },

        "3": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": TEXT_ENCODER_FILE,
                "type": "krea2",
                "device": "default",
            },
        },

        "4": {
            "class_type": "VAELoader",
            "inputs": {
                "vae_name": VAE_FILE,
            },
        },

        "5": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": base_model,
                "weight_dtype": "default",
            },
        },

        "6": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": _ref("5"),
                "lora_name": IDENTITY_COMFY_NAME,
                "strength_model": 1.0,
            },
        },

        "7": {
            "class_type": "VAEEncode",
            "inputs": {
                "pixels": _ref("1"),
                "vae": _ref("4"),
            },
        },

        "8": {
            "class_type": "EmptySD3LatentImage",
            "inputs": {
                "width": DEFAULT_WIDTH,
                "height": DEFAULT_HEIGHT,
                "batch_size": 1,
            },
        },

        "9": {
            "class_type": "Krea2EditModelPatch",
            "inputs": {
                "model": _ref("6"),
                "source_latent": _ref("7"),
                "vae": _ref("4"),
                "source_image": _ref("1"),
                "target_latent": _ref("8"),
                "ref_boost": DEFAULT_REF_BOOST,
                "ref_boost_a": DEFAULT_REF_BOOST,
                "fit_mode": "fit",
            },
        },

        "10": {
            "class_type": "Krea2EditGroundedEncode",
            "inputs": {
                "clip": _ref("3"),
                "image": _ref("1"),
                "prompt": "",
                "grounding_px": DEFAULT_GROUNDING,
            },
        },

        "11": {
            "class_type": "ConditioningZeroOut",
            "inputs": {
                "conditioning": _ref("10"),
            },
        },

        "12": {
            "class_type": "ModelSamplingAuraFlow",
            "inputs": {
                "model": _ref("9"),
                "shift": 4.0,
            },
        },

        "13": {
            "class_type": "KSampler",
            "inputs": {
                "model": _ref("12"),
                "positive": _ref("10"),
                "negative": _ref("11"),
                "latent_image": _ref("8"),
                "seed": DEFAULT_SEED,
                "steps": DEFAULT_STEPS,
                "cfg": DEFAULT_CFG,
                "sampler_name": DEFAULT_SAMPLER,
                "scheduler": DEFAULT_SCHEDULER,
                "denoise": 1.0,
            },
        },

        "14": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": _ref("13"),
                "vae": _ref("4"),
            },
        },

        "15": {
            "class_type": "SaveImage",
            "inputs": {
                "images": _ref("14"),
                "filename_prefix": "krea2_edit",
            },
        },
    }


    if has_second_reference:

        workflow["2"] = {
            "class_type": "LoadImage",
            "inputs": {
                "image": "",
            },
        }

        workflow["16"] = {
            "class_type": "VAEEncode",
            "inputs": {
                "pixels": _ref("2"),
                "vae": _ref("4"),
            },
        }

        workflow["9"]["inputs"][
            "source_latent_b"
        ] = _ref("16")

        workflow["9"]["inputs"][
            "source_image_b"
        ] = _ref("2")

        workflow["10"]["inputs"][
            "image_b"
        ] = _ref("2")


    return workflow


# ============================================================================
# FIND NODE
# ============================================================================

def _find_node(
    workflow: dict[str, Any],
    class_type: str,
) -> str:

    for node_id, node in workflow.items():

        if node.get("class_type") == class_type:

            return node_id

    raise KeyError(
        f"workflow does not contain "
        f"{class_type}"
    )


# ============================================================================
# LORA CHAIN
# ============================================================================

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


    for index, (
        filename,
        strength,
    ) in enumerate(
        enabled_loras
    ):

        node_id = (
            f"user_lora_{index}"
        )


        workflow[node_id] = {

            "class_type": "LoraLoader",

            "inputs": {

                "model": previous_model,

                "clip": previous_clip,

                "lora_name": filename,

                "strength_model": float(
                    strength
                ),

                "strength_clip": float(
                    strength
                ),
            },
        }


        previous_model = _ref(
            node_id
        )

        previous_clip = _ref(
            node_id,
            1,
        )


    for node_id, input_name in (
        model_consumers
    ):

        workflow[node_id]["inputs"][
            input_name
        ] = previous_model


    for node_id, input_name in (
        clip_consumers
    ):

        workflow[node_id]["inputs"][
            input_name
        ] = previous_clip


# ============================================================================
# EDIT IMAGE
# ============================================================================

def _prepare_edit_image(
    path: str,
    target_megapixels: float,
) -> tuple[str, int, int]:

    with Image.open(path) as source:

        image = source.convert("RGB")


        megapixels = max(
            0.25,
            min(
                MAX_TARGET_MP,
                float(target_megapixels),
            ),
        )


        scale = (
            megapixels
            * 1_000_000
            / max(
                1,
                image.width
                * image.height,
            )
        ) ** 0.5


        width = max(
            64,
            int(
                round(
                    image.width
                    * scale
                    / 64
                )
                * 64
            ),
        )


        height = max(
            64,
            int(
                round(
                    image.height
                    * scale
                    / 64
                )
                * 64
            ),
        )


        width = min(
            MAX_WIDTH,
            width,
        )

        height = min(
            MAX_HEIGHT,
            height,
        )


        image = image.resize(
            (width, height),
            Image.Resampling.LANCZOS,
        )


        name = (
            f"input_"
            f"{uuid.uuid4().hex[:12]}"
            f".png"
        )


        image.save(
            INPUT / name,
            format="PNG",
        )


    return (
        name,
        width,
        height,
    )


# ============================================================================
# STAGE IMAGE
# ============================================================================

def _stage_image(
    path: str,
    prefix: str,
) -> str:

    with Image.open(path) as source:

        image = source.convert("RGB")


        name = (
            f"{prefix}_"
            f"{uuid.uuid4().hex[:12]}"
            f".png"
        )


        image.save(
            INPUT / name,
            format="PNG",
        )


    return name


# ============================================================================
# REQUEST VALIDATION
# ============================================================================

def _validate_request(
    mode: str,
    prompt: str,
    edit_prompt: str,
    primary: str | None,
) -> None:

    if mode not in {
        "text2image",
        "edit",
    }:

        raise ValueError(
            "unsupported generation mode"
        )


    if (
        mode == "text2image"
        and not (prompt or "").strip()
    ):

        raise ValueError(
            "enter a prompt"
        )


    if mode == "edit":

        if not primary:

            raise ValueError(
                "upload a primary image "
                "for edit mode"
            )


        if not (
            edit_prompt
            or prompt
            or ""
        ).strip():

            raise ValueError(
                "enter an edit instruction"
            )


# ============================================================================
# T2I INJECTION
# ============================================================================

def _inject_t2i(
    workflow: dict[str, Any],
    *,
    prompt: str,
    width: int,
    height: int,
    steps: int,
    cfg: float,
    sampler: str,
    scheduler: str,
    seed: int,
    enabled_loras: list[
        tuple[str, float]
    ] | None = None,
) -> None:

    _inject_lora_chain(
        workflow,
        enabled_loras or [],
        model_source=_ref("1"),
        clip_source=_ref("2"),
        model_consumers=[
            ("4", "model"),
        ],
        clip_consumers=[
            ("5", "clip"),
        ],
    )


    workflow["5"]["inputs"][
        "text"
    ] = prompt.strip()


    workflow["7"]["inputs"].update(
        width=int(width),
        height=int(height),
    )


    workflow["8"]["inputs"].update(
        seed=int(seed),
        steps=int(steps),
        cfg=float(cfg),
        sampler_name=sampler,
        scheduler=scheduler,
        denoise=1.0,
    )


# ============================================================================
# EDIT INJECTION
# ============================================================================

def _inject_edit(
    workflow: dict[str, Any],
    *,
    primary_name: str,
    second_name: str | None,
    width: int,
    height: int,
    edit_prompt: str,
    grounding_px: int,
    ref_boost: float,
    ref_boost_a: float,
    steps: int,
    cfg: float,
    sampler: str,
    scheduler: str,
    seed: int,
    enabled_loras: list[
        tuple[str, float]
    ] | None = None,
) -> None:

    _inject_lora_chain(
        workflow,
        enabled_loras or [],
        model_source=_ref("6"),
        clip_source=_ref("3"),
        model_consumers=[
            ("9", "model"),
        ],
        clip_consumers=[
            ("10", "clip"),
        ],
    )


    workflow["1"]["inputs"][
        "image"
    ] = primary_name


    workflow["8"]["inputs"].update(
        width=int(width),
        height=int(height),
    )


    workflow["9"]["inputs"].update(
        ref_boost=float(ref_boost),
        ref_boost_a=float(ref_boost_a),
    )


    workflow["10"]["inputs"].update(
        prompt=edit_prompt.strip(),
        grounding_px=int(
            grounding_px
        ),
    )


    workflow["13"]["inputs"].update(
        seed=int(seed),
        steps=int(steps),
        cfg=float(cfg),
        sampler_name=sampler,
        scheduler=scheduler,
        denoise=1.0,
    )


    if second_name:

        workflow["2"]["inputs"][
            "image"
        ] = second_name


# ============================================================================
# EXECUTE
# ============================================================================

def _execute_workflow(
    workflow: dict[str, Any],
) -> list[str]:

    import execution
    import server
    from app.assets.manager import default_asset_manager
    loop = asyncio.new_event_loop()

    asyncio.set_event_loop(loop)


    server_instance = server.PromptServer(
        loop,
        default_asset_manager()
    )


    executor = execution.PromptExecutor(
        server_instance,
        cache_type=execution.CacheType.RAM_PRESSURE,
        cache_args={
            "lru": 0,
            "ram": 2.0,
            "ram_inactive": 8.0,
        },
    )


    prompt_id = str(uuid.uuid4())


    save_id = _find_node(
        workflow,
        "SaveImage",
    )


    executor.execute(
        workflow,
        prompt_id,
        extra_data={},
        execute_outputs=[
            save_id
        ],
    )


    if not executor.success:

        message = (
            executor.status_messages[-1]
            if executor.status_messages
            else "ComfyUI execution failed"
        )

        raise RuntimeError(
            str(message)
        )


    paths: list[pathlib.Path] = []


    for output in (
        executor.history_result
        .get("outputs", {})
        .values()
    ):

        for items in output.values():

            if not isinstance(
                items,
                list,
            ):

                continue


            for item in items:

                if (
                    not isinstance(
                        item,
                        dict,
                    )
                    or not item.get(
                        "filename"
                    )
                ):

                    continue


                base = (
                    OUTPUT
                    if item.get(
                        "type",
                        "output",
                    )
                    == "output"
                    else COMFY
                    / item.get(
                        "type",
                        "output",
                    )
                )


                candidate = (
                    base
                    / item.get(
                        "subfolder",
                        "",
                    )
                    / item["filename"]
                )


                if candidate.exists():

                    paths.append(
                        candidate
                    )


    if not paths:

        paths = sorted(
            [
                pathlib.Path(item)
                for item in glob.glob(
                    str(
                        OUTPUT
                        / "**"
                        / "*.png"
                    ),
                    recursive=True,
                )
            ],
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )


    if not paths:

        raise RuntimeError(
            "ComfyUI finished without "
            "an output image"
        )


    return [
        str(path)
        for path in paths
    ]


# ============================================================================
# RUNTIME
# ============================================================================

def _prepare_runtime(
    base_model: str,
    progress: gr.Progress | None = None,
) -> str:

    _ensure_comfy()

    _scan_local_models()

    _validate_required_assets()


    if not LOCAL_BASE_MODELS:

        raise RuntimeError(
            "No diffusion models found in:\n"
            f"{DIFFUSION_DIR}\n\n"
            "Put your .safetensors diffusion "
            "model inside that directory."
        )


    resolved_base_model = (
        _validate_model_name(
            base_model
        )
    )


    _init_comfy_nodes()


    return resolved_base_model


# ============================================================================
# GPU TIME
# ============================================================================

def get_gpu_duration(
    *args: Any,
    **kwargs: Any,
) -> int:

    steps = kwargs.get(
        "steps",
        args[11]
        if len(args) > 11
        else DEFAULT_STEPS,
    )


    width = kwargs.get(
        "width",
        args[5]
        if len(args) > 5
        else DEFAULT_WIDTH,
    )


    height = kwargs.get(
        "height",
        args[6]
        if len(args) > 6
        else DEFAULT_HEIGHT,
    )


    gen_budget = kwargs.get(
        "gen_budget",
        args[17]
        if len(args) > 17
        else 0,
    )


    if gen_budget and int(
        gen_budget
    ) > 0:

        return max(
            MIN_GPU_SECONDS,
            min(
                MAX_GPU_SECONDS,
                int(gen_budget),
            ),
        )


    lora_weights = kwargs.get(
        "lora_weights",
        args[19]
        if len(args) > 19
        else {},
    ) or {}


    lora_count = sum(
        1
        for value
        in lora_weights.values()
        if value
        and abs(
            float(value)
        ) > 1e-6
    )


    estimate = int(
        35
        + (
            int(width)
            * int(height)
            / 1_000_000
        )
        * int(steps)
        * 3.0
        * (
            1
            + 0.05
            * lora_count
        )
    )


    return max(
        MIN_GPU_SECONDS,
        min(
            MAX_GPU_SECONDS,
            estimate,
        ),
    )


# ============================================================================
# GENERATE
# ============================================================================

@spaces.GPU(
    duration=get_gpu_duration
)
def generate(
    mode: str,
    prompt: str,
    edit_prompt: str,
    primary_image: str | None,
    second_image: str | None,
    width: int,
    height: int,
    target_megapixels: float,
    grounding_px: int,
    ref_boost: float,
    ref_boost_a: float,
    steps: int,
    cfg: float,
    sampler: str,
    scheduler: str,
    seed: int,
    randomize_seed: bool,
    gen_budget: float,
    base_model: str,
    lora_weights: dict[
        str,
        float
    ] | None = None,
    progress: gr.Progress = gr.Progress(
        track_tqdm=True
    ),
) -> tuple[
    list[str],
    str,
    int,
]:

    effective_seed = (
        random.randint(
            0,
            2**32 - 1,
        )
        if (
            randomize_seed
            or int(seed) < 0
        )
        else int(seed)
    )


    staged: list[pathlib.Path] = []

    total_start = time.time()

    try:

        _validate_request(
            mode,
            prompt,
            edit_prompt,
            primary_image,
        )


        effective_edit_prompt = (
            edit_prompt
            or prompt
            or ""
        ).strip()


        if sampler not in SAMPLERS:

            raise ValueError(
                "unsupported sampler"
            )


        if scheduler not in SCHEDULERS:

            raise ValueError(
                "unsupported scheduler"
            )


        resolved_base_model = (
            _prepare_runtime(
                base_model,
                progress,
            )
        )


        enabled_loras: list[
            tuple[str, float]
        ] = []


        for filename, weight in (
            lora_weights or {}
        ).items():

            if filename not in LOCAL_LORAS:

                raise ValueError(
                    "local LoRA is not available: "
                    + str(filename)
                )


            numeric_weight = float(
                weight
            )


            if (
                numeric_weight < -3.0
                or numeric_weight > 3.0
            ):

                raise ValueError(
                    "LoRA weight out of range: "
                    + filename
                )


            if abs(
                numeric_weight
            ) > 1e-6:

                validated_name = (
                    _validate_lora_name(
                        filename
                    )
                )


                enabled_loras.append(
                    (
                        validated_name,
                        numeric_weight,
                    )
                )


        # --------------------------------------------------------------------
        # T2I
        # --------------------------------------------------------------------

        if mode == "text2image":

            width = max(
                512,
                min(
                    MAX_WIDTH,
                    int(width)
                    // 64
                    * 64,
                ),
            )


            height = max(
                512,
                min(
                    MAX_HEIGHT,
                    int(height)
                    // 64
                    * 64,
                ),
            )


            workflow = _t2i_workflow(
                resolved_base_model
            )


        # --------------------------------------------------------------------
        # EDIT
        # --------------------------------------------------------------------

        else:

            (
                primary_name,
                width,
                height,
            ) = _prepare_edit_image(
                primary_image,
                target_megapixels,
            )


            staged.append(
                INPUT / primary_name
            )


            second_name = (
                _stage_image(
                    second_image,
                    "reference",
                )
                if second_image
                else None
            )


            if second_name:

                staged.append(
                    INPUT / second_name
                )


            workflow = _edit_workflow(
                bool(second_name),
                resolved_base_model,
            )


        # --------------------------------------------------------------------
        # INJECT
        # --------------------------------------------------------------------

        if mode == "text2image":

            _inject_t2i(
                workflow,
                prompt=prompt,
                width=width,
                height=height,
                steps=int(steps),
                cfg=float(cfg),
                sampler=sampler,
                scheduler=scheduler,
                seed=effective_seed,
                enabled_loras=enabled_loras,
            )

        else:

            _inject_edit(
                workflow,
                primary_name=primary_name,
                second_name=second_name,
                width=width,
                height=height,
                edit_prompt=effective_edit_prompt,
                grounding_px=int(
                    grounding_px
                ),
                ref_boost=float(
                    ref_boost
                ),
                ref_boost_a=float(
                    ref_boost_a
                ),
                steps=int(steps),
                cfg=float(cfg),
                sampler=sampler,
                scheduler=scheduler,
                seed=effective_seed,
                enabled_loras=enabled_loras,
            )


        # --------------------------------------------------------------------
        # METADATA
        # --------------------------------------------------------------------

        active_loras = [
            {
                "hf_filename": filename,
                "weight": float(weight),
            }
            for filename, weight
            in enabled_loras
        ]


        settings = build_settings(
            mode=mode,
            prompt=prompt,
            edit_prompt=effective_edit_prompt,
            width=width,
            height=height,
            target_megapixels=float(
                target_megapixels
            ),
            grounding_px=int(
                grounding_px
            ),
            ref_boost=float(
                ref_boost
            ),
            ref_boost_a=float(
                ref_boost_a
            ),
            steps=int(steps),
            cfg=float(cfg),
            sampler_name=sampler,
            scheduler=scheduler,
            seed=int(seed),
            randomize_seed=bool(
                randomize_seed
            ),
            gen_budget=float(
                gen_budget
            ),
            effective_seed=effective_seed,
            base_model=base_model,
            custom_base_model=None,
            catalog_loras=active_loras,
            custom_loras=[],
        )

        t0 = time.time()
        progress(
            0.35,
            desc=f"generating {mode}",
        )


        result_paths = _execute_workflow(
            workflow
        )


        destination_dir = pathlib.Path(
            tempfile.mkdtemp(
                prefix="krea2_outputs_"
            )
        )


        output_paths: list[str] = []


        for index, source in enumerate(
            result_paths
        ):

            destination = (
                destination_dir
                / f"output_{index}.png"
            )


            write_png_metadata(
                source,
                destination,
                settings,
            )


            output_paths.append(
                str(destination)
            )

        # Upload the final metadata-preserving images to MEGA.
        mega_results: list[str] = []
        for output_path in output_paths:
            mega_results.append(_upload_to_mega(output_path, account=MEGA_ACCOUNT))

        print(f"[mega] uploaded {len(mega_results)} image(s)", flush=True)
        print(f"⏱️ Total: "f"{time.time() - total_start:.1f}s")
        return (
            output_paths,
            (
                f"done — "
                f"{len(output_paths)} image(s), "
                f"uploaded to MEGA folder '{os.environ.get('MEGA_FOLDER', 'Krea2-Outputs')}' — "
                f"seed {effective_seed}"
            ),
            effective_seed,
        )


    except Exception as exc:

        print(
            traceback.format_exc(),
            flush=True,
        )


        raise gr.Error(
            "generation failed: "
            + str(exc)[:500]
        ) from exc


    finally:

        for path in staged:

            try:

                path.unlink(
                    missing_ok=True
                )

            except OSError:

                pass


# ============================================================================
# PROFILE
# ============================================================================

def _profile_from_values(
    mode: str,
    prompt: str,
    edit_prompt: str,
    width: int,
    height: int,
    target_mp: float,
    grounding: int,
    ref_boost: float,
    ref_boost_a: float,
    steps: int,
    cfg: float,
    sampler: str,
    scheduler: str,
    seed: int,
    randomize: bool,
    budget: float,
    base_model: str,
    catalog_loras: list[
        dict[str, Any]
    ] | None = None,
) -> dict[str, Any]:

    return build_settings(
        mode=mode,
        prompt=prompt,
        edit_prompt=edit_prompt,
        width=int(width),
        height=int(height),
        target_megapixels=float(
            target_mp
        ),
        grounding_px=int(
            grounding
        ),
        ref_boost=float(
            ref_boost
        ),
        ref_boost_a=float(
            ref_boost_a
        ),
        steps=int(steps),
        cfg=float(cfg),
        sampler_name=sampler,
        scheduler=scheduler,
        seed=int(seed),
        randomize_seed=bool(
            randomize
        ),
        gen_budget=float(
            budget
        ),
        effective_seed=int(
            seed
        ),
        base_model=base_model,
        custom_base_model=None,
        catalog_loras=catalog_loras,
        custom_loras=[],
    )


# ============================================================================
# UI
# ============================================================================

def create_ui() -> gr.Blocks:

    with gr.Blocks(
        title="Krea 2 Turbo Image Generator",
        theme=gr.themes.Soft(),
    ) as demo:

        with gr.Row():

            with gr.Column(
                scale=1
            ):

                mode = gr.Radio(
                    [
                        "text2image",
                        "edit",
                    ],
                    value="text2image",
                    label="mode",
                )


                base_model = gr.Dropdown(
                    choices=LOCAL_BASE_MODELS,
                    value=(
                        LOCAL_BASE_MODELS[0]
                        if LOCAL_BASE_MODELS
                        else None
                    ),
                    label="base diffusion model",
                    allow_custom_value=False,
                )


                gr.Markdown(
                    "Models are loaded from "
                    f"`{DIFFUSION_DIR}`"
                )


                with gr.Column(
                    visible=False
                ) as image_inputs:

                    primary = gr.Image(
                        type="filepath",
                        label="primary image / scene",
                    )

                    second = gr.Image(
                        type="filepath",
                        label="optional second reference",
                    )


                prompt = gr.Textbox(
                    value=(
                        "A cinematic portrait "
                        "in soft natural light"
                    ),
                    label="prompt",
                    lines=3,
                )


                edit_prompt = gr.Textbox(
                    label="edit instruction",
                    lines=3,
                    visible=False,
                    placeholder=(
                        "recolor the jacket "
                        "to matte black"
                    ),
                )


                with gr.Column() as t2i_resolution:

                    with gr.Row():

                        width = gr.Slider(
                            512,
                            MAX_WIDTH,
                            value=DEFAULT_WIDTH,
                            step=64,
                            label="width",
                        )

                        height = gr.Slider(
                            512,
                            MAX_HEIGHT,
                            value=DEFAULT_HEIGHT,
                            step=64,
                            label="height",
                        )


                with gr.Column(
                    visible=False
                ) as edit_controls:

                    target_mp = gr.Slider(
                        0.25,
                        MAX_TARGET_MP,
                        value=DEFAULT_TARGET_MP,
                        step=0.05,
                        label="target megapixels",
                    )

                    grounding = gr.Slider(
                        384,
                        1536,
                        value=DEFAULT_GROUNDING,
                        step=32,
                        label="grounding resolution",
                    )

                    ref_boost = gr.Slider(
                        0.0,
                        12.0,
                        value=DEFAULT_REF_BOOST,
                        step=0.1,
                        label="primary reference strength",
                    )

                    ref_boost_a = gr.Slider(
                        0.0,
                        12.0,
                        value=DEFAULT_REF_BOOST,
                        step=0.1,
                        label="second reference strength",
                    )


                # ----------------------------------------------------------------
                # LORAS
                # ----------------------------------------------------------------

                with gr.Accordion(
                    f"Local LoRAs "
                    f"({len(LOCAL_LORAS)} available)",
                    open=False,
                ):

                    gr.Markdown(
                        "LoRAs are loaded from "
                        f"`{LORA_ROOT}`. "
                        "Set any LoRA weight above zero "
                        "or below zero to enable it. "
                        "Multiple LoRAs can be used together."
                    )


                    if not LOCAL_LORAS:

                        gr.Markdown(
                            "⚠️ No LoRAs found."
                        )


                    lora_slider_map: dict[
                        str,
                        gr.Slider,
                    ] = {}


                    for filename in LOCAL_LORAS:

                        lora_slider_map[
                            filename
                        ] = gr.Slider(
                            minimum=-3.0,
                            maximum=3.0,
                            value=0.0,
                            step=0.05,
                            label=filename,
                        )


                with gr.Accordion(
                    "sampling",
                    open=False,
                ):

                    steps = gr.Slider(
                        4,
                        40,
                        value=DEFAULT_STEPS,
                        step=1,
                        label="steps",
                    )


                    cfg = gr.Slider(
                        1.0,
                        5.0,
                        value=DEFAULT_CFG,
                        step=0.1,
                        label="CFG",
                    )


                    with gr.Row():

                        sampler = gr.Dropdown(
                            SAMPLERS,
                            value=DEFAULT_SAMPLER,
                            label="sampler",
                        )

                        scheduler = gr.Dropdown(
                            SCHEDULERS,
                            value=DEFAULT_SCHEDULER,
                            label="scheduler",
                        )


                with gr.Row():

                    seed = gr.Number(
                        value=DEFAULT_SEED,
                        precision=0,
                        label="seed",
                    )

                    randomize = gr.Checkbox(
                        value=False,
                        label="randomize seed",
                    )


                gen_budget = gr.Slider(
                    0,
                    MAX_GPU_SECONDS,
                    value=0,
                    step=10,
                    label="GPU budget (0 = automatic)",
                )


                button = gr.Button(
                    "generate",
                    variant="primary",
                    size="lg",
                )


            with gr.Column(
                scale=1
            ):

                gallery = gr.Gallery(
                    label="output",
                    columns=2,
                    height=600,
                )

                status = gr.Textbox(
                    label="status",
                    interactive=False,
                )

                used_seed = gr.Number(
                    label="used seed",
                    interactive=False,
                )


        # ====================================================================
        # MODE
        # ====================================================================

        def on_mode_change(
            value: str,
        ):

            editing = (
                value == "edit"
            )

            return (
                gr.update(
                    visible=editing
                ),
                gr.update(
                    visible=not editing
                ),
                gr.update(
                    visible=editing
                ),
                gr.update(
                    visible=editing
                ),
            )


        mode.change(
            on_mode_change,
            inputs=[mode],
            outputs=[
                image_inputs,
                t2i_resolution,
                edit_controls,
                edit_prompt,
            ],
        )


        # ====================================================================
        # LORA
        # ====================================================================

        all_lora_filenames = list(
            lora_slider_map.keys()
        )

        all_lora_sliders = list(
            lora_slider_map.values()
        )


        def _catalog_weights(
            values: list[Any],
        ) -> dict[str, float]:

            return {
                filename: float(weight)
                for filename, weight
                in zip(
                    all_lora_filenames,
                    values,
                )
                if (
                    weight
                    and abs(
                        float(weight)
                    ) > 1e-6
                )
            }


        # ====================================================================
        # GENERATION WRAPPER
        # ====================================================================

        def _generate_wrapper(
            *values,
        ):

            base_values = values[:18]

            base_model_value = values[18]

            lora_values = values[19:]


            lora_weights = (
                _catalog_weights(
                    list(lora_values)
                )
            )


            return generate(
                *base_values,
                base_model=base_model_value,
                lora_weights=lora_weights,
            )


        generation_inputs = [

            mode,
            prompt,
            edit_prompt,
            primary,
            second,

            width,
            height,

            target_mp,

            grounding,
            ref_boost,
            ref_boost_a,

            steps,
            cfg,

            sampler,
            scheduler,

            seed,
            randomize,

            gen_budget,

            base_model,

            *all_lora_sliders,
        ]


        button.click(
            _generate_wrapper,
            inputs=generation_inputs,
            outputs=[
                gallery,
                status,
                used_seed,
            ],
        )


    return demo


# ============================================================================
# STARTUP
# ============================================================================

def _on_startup() -> None:

    if (
        os.environ.get(
            "KREA_SKIP_STARTUP"
        )
        == "1"
    ):

        return


    try:

        print(
            "=" * 70,
            flush=True,
        )

        print(
            "Krea 2 Turbo Starting",
            flush=True,
        )

        print(
            "=" * 70,
            flush=True,
        )


        print(
            "[startup] ROOT:",
            ROOT,
            flush=True,
        )

        print(
            "[startup] COMFY:",
            COMFY,
            flush=True,
        )

        print(
            "[startup] MODELS:",
            MODELS,
            flush=True,
        )


        _ensure_comfy()


        _scan_local_models()


        try:

            _validate_required_assets()

        except Exception as exc:

            print(
                "[startup] model warning:",
                str(exc),
                flush=True,
            )


        _init_comfy_nodes()


        print(
            "[startup] ready",
            flush=True,
        )


    except Exception as exc:

        print(
            "[startup] setup incomplete "
            f"({type(exc).__name__}: {exc})",
            flush=True,
        )

        print(
            "[startup] generation will retry setup",
            flush=True,
        )


# ============================================================================
# MAIN
# ============================================================================

_on_startup()

_scan_local_models()

demo = create_ui()

demo.queue()


if __name__ == "__main__":

    demo.launch(
        share=True,
    )
