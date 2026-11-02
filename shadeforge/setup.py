"""One-command local backend setup: install ComfyUI, pin a working stack, fetch a model, start it.

Everything here is hard-won from a real install on an Intel Mac. The pins are not arbitrary:
PyTorch stopped shipping macOS x86_64 wheels after 2.2.2, so an Intel Mac needs a 2024-era
ComfyUI to match. Apple Silicon and Linux get the current stack instead.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import struct
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

COMFY_DIR = Path(os.environ.get("COMFYUI_DIR", Path.home() / "ComfyUI"))
COMFY_REPO = "https://github.com/comfyanonymous/ComfyUI"

# Last ComfyUI commit from the torch-2.2.2 era. Only used on Intel macOS.
LEGACY_PIN_BEFORE = "2024-04-01"
LEGACY_TORCH = ["torch==2.2.2", "torchvision==0.17.2", "numpy<2"]
MODERN_TORCH = ["torch", "torchvision"]

# Checkpoint catalogue, best first. `gb` is the download size; `quality` ranks
# prompt adherence and photorealism as observed on real runs, not marketing claims.
MODELS = {
    "realistic-vision": {
        "file": "realisticVision_v51.safetensors",
        "url": ("https://huggingface.co/SG161222/Realistic_Vision_V5.1_noVAE/"
                "resolve/main/Realistic_Vision_V5.1_fp16-no-ema.safetensors"),
        "gb": 2.0,
        "quality": 2,
        "license": "CreativeML OpenRAIL-M (commercial use allowed)",
        "note": "Photoreal fine-tune. Much better products, people and lighting than SD1.5.",
    },
    "sd15": {
        "file": "v1-5-pruned-emaonly.safetensors",
        "url": ("https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/"
                "resolve/main/v1-5-pruned-emaonly.safetensors"),
        "gb": 4.0,
        "quality": 1,
        "license": "CreativeML OpenRAIL-M (commercial use allowed)",
        "note": "Base model. Generic scenes only; weak on specific subjects.",
    },
}
DEFAULT_MODEL = "realistic-vision"

# Ranked preference when auto-selecting among already-installed checkpoints.
CHECKPOINT_PREFERENCE = ["realisticvision", "realistic_vision", "epicrealism", "dreamshaper",
                         "sdxl", "v1-5", "sd15"]

# SD1.5 is trained at 512px. Past ~768 on an axis it duplicates subjects (two carts, extra limbs).
SD15_MAX_EDGE = 768

DISK_HEADROOM_GB = 8


class SetupError(RuntimeError):
    pass


def is_legacy_mac() -> bool:
    """Intel macOS: no MPS, and no PyTorch wheels past 2.2.2."""
    return sys.platform == "darwin" and platform.machine() == "x86_64"


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1024**3


def find_python(version: str = "3.11") -> str | None:
    """Locate an interpreter torch can actually install into."""
    candidates = [
        shutil.which(f"python{version}"),
        str(Path.home() / f".local/bin/python{version}"),          # uv-managed
        f"/usr/local/opt/python@{version}/bin/python{version}",     # brew intel
        f"/opt/homebrew/opt/python@{version}/bin/python{version}",  # brew arm
        str(Path.home() / f".pyenv/versions/{version}/bin/python"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return None


def run(cmd: list[str], cwd: Path | None = None, quiet: bool = False) -> None:
    result = subprocess.run(cmd, cwd=str(cwd) if cwd else None,
                            capture_output=quiet, text=True)
    if result.returncode != 0:
        tail = (result.stderr or result.stdout or "")[-600:] if quiet else ""
        raise SetupError(f"Command failed ({' '.join(cmd[:3])}...): {tail}")


def verify_safetensors(path: Path) -> bool:
    """A truncated download still looks like a checkpoint to ComfyUI. Check the header math."""
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            header_len = struct.unpack("<Q", handle.read(8))[0]
            # A junk file's first 8 bytes read as an enormous length; allocating that
            # blows up with MemoryError instead of returning False.
            if header_len > min(size, 100 * 1024**2):
                return False
            header = json.loads(handle.read(header_len))
        end = max(v["data_offsets"][1] for k, v in header.items() if k != "__metadata__")
        return 8 + header_len + end == size
    except (OSError, ValueError, KeyError, TypeError, MemoryError, struct.error):
        return False


def comfy_python() -> Path:
    return COMFY_DIR / ".venv" / "bin" / "python"


def is_installed() -> bool:
    return comfy_python().exists() and (COMFY_DIR / "main.py").exists()


def has_model() -> bool:
    directory = COMFY_DIR / "models" / "checkpoints"
    return any(p.suffix in (".safetensors", ".ckpt") for p in directory.glob("*")) \
        if directory.exists() else False


def pick_model(free: float | None = None) -> str | None:
    """Best model that fits the disk, with headroom for outputs and the venv.

    Returns None when nothing fits -- the caller must then refuse rather than
    start a download that fills the disk.
    """
    free = free_gb(Path.home()) if free is None else free
    affordable = [(spec["quality"], key) for key, spec in MODELS.items()
                  if free >= spec["gb"] + 3]
    return max(affordable)[1] if affordable else None


def report_disk(model_key: str | None = None) -> None:
    free = free_gb(Path.home())
    print(f"Disk free: {free:.1f} GB")
    for key, spec in sorted(MODELS.items(), key=lambda kv: -kv[1]["quality"]):
        fits = "fits" if free >= spec["gb"] + 3 else "NOT enough space"
        mark = " <- selected" if key == model_key else ""
        print(f"  {key:<17} {spec['gb']:>4.1f} GB  {fits}{mark}")
        print(f"  {'':<17} {spec['note']}")


def install(skip_model: bool = False, model: str = "auto") -> None:
    legacy = is_legacy_mac()
    print(f"ShadeForge setup -> {COMFY_DIR}")
    print(f"Platform: {platform.machine()} {'(Intel Mac: pinning 2024-era stack)' if legacy else ''}")

    chosen: str | None = None
    if not skip_model and not has_model():
        chosen = pick_model() if model == "auto" else model
        if model != "auto" and model not in MODELS:
            raise SetupError(f"Unknown model '{model}'. Choose from: {', '.join(MODELS)}")
        report_disk(chosen)
        if chosen is None:
            raise SetupError(
                f"Not enough disk for any model ({free_gb(Path.home()):.1f} GB free; the "
                "smallest needs ~5 GB with headroom). Free space and re-run `shadeforge setup`."
            )
        if free_gb(Path.home()) < MODELS[chosen]["gb"] + 3:
            raise SetupError(f"'{chosen}' needs ~{MODELS[chosen]['gb'] + 3:.0f} GB free.")

    needed = 3 if (skip_model or has_model()) else MODELS[chosen]["gb"] + 3
    if free_gb(Path.home()) < needed:
        raise SetupError(
            f"Only {free_gb(Path.home()):.1f} GB free; need ~{needed:.0f} GB. "
            "Clear space first (Xcode iOS DeviceSupport and Android system-images are usually "
            "the biggest easy wins)."
        )

    if not (COMFY_DIR / "main.py").exists():
        print("Cloning ComfyUI...")
        run(["git", "clone", "--quiet", COMFY_REPO, str(COMFY_DIR)])
        if legacy:
            pin = subprocess.run(["git", "rev-list", "-1", f"--before={LEGACY_PIN_BEFORE}", "master"],
                                 cwd=COMFY_DIR, capture_output=True, text=True).stdout.strip()
            if pin:
                run(["git", "checkout", "--quiet", pin], cwd=COMFY_DIR)
                print(f"Pinned ComfyUI to {pin[:10]} (matches torch 2.2.2).")
    else:
        print("ComfyUI already present.")

    if not comfy_python().exists():
        interpreter = find_python("3.11") if legacy else (find_python("3.12") or sys.executable)
        if not interpreter:
            raise SetupError(
                "Need Python 3.11 for torch 2.2.2 on Intel macOS, and none was found. "
                "Install one with: uv python install 3.11"
            )
        print(f"Creating venv with {interpreter}...")
        run([interpreter, "-m", "venv", str(COMFY_DIR / ".venv")])
        run([str(comfy_python()), "-m", "pip", "install", "-q", "--upgrade", "pip"], quiet=True)

    try:
        run([str(comfy_python()), "-c", "import torch"], quiet=True)
        print("torch already installed.")
    except SetupError:
        pins = LEGACY_TORCH if legacy else MODERN_TORCH
        print(f"Installing {', '.join(pins)} (large download)...")
        run([str(comfy_python()), "-m", "pip", "install", *pins])

    print("Installing ComfyUI requirements...")
    run([str(comfy_python()), "-m", "pip", "install", "-q", "-r",
         str(COMFY_DIR / "requirements.txt")])
    if legacy:
        # requirements.txt can drag numpy 2.x back in; torch 2.2.2 was built against 1.x
        # and segfaults on the array bridge otherwise.
        run([str(comfy_python()), "-m", "pip", "install", "-q", "numpy<2"], quiet=True)

    if skip_model or has_model():
        print("Checkpoint already present." if has_model() else "Skipping model download.")
    else:
        spec = MODELS[chosen]
        target = COMFY_DIR / "models" / "checkpoints" / spec["file"]
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {chosen} ({spec['gb']} GB, resumable). License: {spec['license']}")
        run(["curl", "-L", "--fail", "--retry", "3", "-C", "-", "-o", str(target), spec["url"]])
        if not verify_safetensors(target):
            target.unlink(missing_ok=True)
            raise SetupError("Checkpoint failed integrity check and was removed. Re-run setup.")
        print("Checkpoint verified intact.")

    print("Setup complete.")


def server_ready(url: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(f"{url}/object_info/CheckpointLoaderSimple", timeout=timeout):
            return True
    except Exception:
        return False


def start_server(url: str, wait: int = 120) -> None:
    if server_ready(url):
        return
    if not is_installed():
        raise SetupError("ComfyUI is not installed. Run: shadeforge setup")
    port = url.rsplit(":", 1)[-1] or "8188"
    log = COMFY_DIR / "comfyui.log"
    cmd = [str(comfy_python()), "main.py", "--port", port]
    if is_legacy_mac():
        cmd.append("--cpu")  # Intel Macs have no Metal/MPS backend
    print(f"Starting ComfyUI (log: {log})...")
    with log.open("ab") as handle:
        subprocess.Popen(cmd, cwd=str(COMFY_DIR), stdout=handle, stderr=handle,
                         start_new_session=True)
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        if server_ready(url):
            print("ComfyUI is up.")
            return
        time.sleep(2)
    raise SetupError(f"ComfyUI did not become ready within {wait}s. Check {log}.")


def ensure_local(url: str, auto: bool = True, allow_install: bool = True) -> None:
    """Make local generation possible: install if missing, start if stopped.

    allow_install guards the ~4 GB first-run download. Starting an already-installed
    server is cheap and always allowed; a full install is not something to trigger by
    surprise when a cloud fallback was on the table.
    """
    if server_ready(url):
        return
    if not auto:
        raise SetupError("Local backend is not running. Start it with: shadeforge start")
    if not is_installed() or not has_model():
        if not allow_install:
            raise SetupError("Local backend is not installed (run: shadeforge setup).")
        print("Local backend not installed yet -- setting it up (one time, ~4 GB download).")
        install()
    start_server(url)
