from __future__ import annotations

import argparse
import base64
import json
import os
import platform
import random
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

from . import setup as setup_mod
from . import typeset

__version__ = "0.4.0"

DEFAULT_COMFY_URL = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188").rstrip("/")
DEFAULT_GEMINI_MODEL = os.environ.get("GEMINI_IMAGE_MODEL", "gemini-3-pro-image-preview")
USER_AGENT = f"ShadeForge/{__version__}"


class ShadeForgeError(RuntimeError):
    pass


def request_json(url: str, method: str = "GET", data: dict | None = None,
                 headers: dict | None = None, timeout: int = 20) -> Any:
    body = json.dumps(data).encode("utf-8") if data is not None else None
    req_headers = {"User-Agent": USER_AGENT}
    if body is not None:
        req_headers["Content-Type"] = "application/json"
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(url, data=body, headers=req_headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read()
        return json.loads(raw.decode("utf-8")) if raw else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1200]
        raise ShadeForgeError(f"HTTP {exc.code} from {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise ShadeForgeError(f"Could not reach {url}: {exc.reason}") from exc
    except TimeoutError as exc:
        raise ShadeForgeError(f"Request timed out while contacting {url}") from exc
    except json.JSONDecodeError as exc:
        raise ShadeForgeError(f"{url} returned a non-JSON response") from exc


def get_gemini_key() -> str | None:
    """Environment variables only. Never scans files, history or keychains for secrets."""
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or None


def _names_from_object_info(payload: Any) -> list[str]:
    """ComfyUI /object_info/CheckpointLoaderSimple -> required.ckpt_name[0] is the name list."""
    try:
        spec = payload["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0]
    except (KeyError, IndexError, TypeError):
        return []
    return [str(n) for n in spec if isinstance(n, str)] if isinstance(spec, list) else []


def list_checkpoints(url: str = DEFAULT_COMFY_URL, timeout: int = 5) -> list[str]:
    names = _names_from_object_info(
        request_json(f"{url}/object_info/CheckpointLoaderSimple", timeout=timeout)
    )
    if names:
        return names
    # Fallback for ComfyUI builds exposing /models/<folder> as a list of filenames.
    # Reachability was already proven above, so a failure here just means "no models",
    # not "server down" -- don't let it masquerade as a connection error.
    try:
        payload = request_json(f"{url}/models/checkpoints", timeout=timeout)
    except ShadeForgeError:
        return []
    if isinstance(payload, list):
        return [p if isinstance(p, str) else str(p.get("name", "")) for p in payload
                if isinstance(p, str) or (isinstance(p, dict) and p.get("name"))]
    return []


def check_comfy(url: str = DEFAULT_COMFY_URL) -> tuple[bool, list[str], str | None]:
    try:
        return True, list_checkpoints(url), None
    except ShadeForgeError as exc:
        return False, [], str(exc)


def doctor(url: str = DEFAULT_COMFY_URL) -> int:
    print(f"ShadeForge {__version__} — local-first image generation")
    print(f"Platform: {platform.platform()}")
    print(f"ComfyUI URL: {url}")
    ok, models, error = check_comfy(url)
    if ok:
        print("ComfyUI: reachable")
        if models:
            print(f"Checkpoint models: {len(models)}")
            for name in models[:8]:
                print(f"  - {name}")
            if len(models) > 8:
                print(f"  ... and {len(models) - 8} more")
        else:
            print("Checkpoint models: none found")
            print("  Add a checkpoint to ComfyUI/models/checkpoints, then refresh ComfyUI.")
    else:
        print("ComfyUI: not reachable")
        print("  Start ComfyUI, then retry. Default URL is http://127.0.0.1:8188.")
        print(f"  Detail: {error}")
    print(f"Gemini API key: {'available (value hidden)' if get_gemini_key() else 'not detected'}")
    print("Privacy: auto mode prefers local. Cloud is never used unless explicitly requested.")
    return 0 if (ok and models) else 1


def _node(class_type: str, inputs: dict) -> dict:
    return {"class_type": class_type, "inputs": inputs}


def build_comfy_workflow(prompt: str, checkpoint: str, width: int, height: int, steps: int,
                         seed: int, negative_prompt: str = "", cfg: float = 7.0) -> dict:
    return {
        "1": _node("CheckpointLoaderSimple", {"ckpt_name": checkpoint}),
        "2": _node("CLIPTextEncode", {"text": prompt, "clip": ["1", 1]}),
        "3": _node("CLIPTextEncode", {"text": negative_prompt, "clip": ["1", 1]}),
        "4": _node("EmptyLatentImage", {"width": width, "height": height, "batch_size": 1}),
        "5": _node("KSampler", {
            "seed": seed, "steps": steps, "cfg": cfg, "sampler_name": "euler",
            "scheduler": "normal", "denoise": 1.0, "model": ["1", 0],
            "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["4", 0],
        }),
        "6": _node("VAEDecode", {"samples": ["5", 0], "vae": ["1", 2]}),
        "7": _node("SaveImage", {"filename_prefix": "shadeforge", "images": ["6", 0]}),
    }


def _download_comfy_image(url: str, image_info: dict, destination: Path) -> None:
    query = urllib.parse.urlencode({
        "filename": image_info["filename"],
        "subfolder": image_info.get("subfolder", ""),
        "type": image_info.get("type", "output"),
    })
    req = urllib.request.Request(f"{url}/view?{query}", headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            content = response.read()
    except urllib.error.URLError as exc:
        raise ShadeForgeError(f"Could not download image from ComfyUI: {exc}") from exc
    if not content:
        raise ShadeForgeError("ComfyUI returned an empty image.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)


# ComfyUI's sampler logs a tqdm bar: " 65%|######5   | 13/20 [03:19<01:44, 14.90s/it]"
PROGRESS_RE = re.compile(r"(\d+)/(\d+)\s*\[[\d:]+<([\d:?]+)")

STATE_FILE = Path.home() / ".shadeforge" / "progress.json"


def write_progress_state(log_path: Path, start: int) -> None:
    """Record which Claude Code session owns this run.

    The statusline is global -- every open session runs the same script. Without
    an owner recorded here, one generation would paint a progress bar across
    every terminal the user has open.
    """
    try:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps({
            "session": os.environ.get("CLAUDE_CODE_SESSION_ID", ""),
            "log": str(log_path),
            "start": start,
            "pid": os.getpid(),
            "started": time.time(),
        }))
    except OSError:
        pass  # progress display is a nicety; never fail a generation over it


def clear_progress_state() -> None:
    try:
        STATE_FILE.unlink(missing_ok=True)
    except OSError:
        pass


def read_comfy_progress(log_path: Path, start_pos: int = 0) -> tuple[int, int, str] | None:
    """Last sampler step reported since start_pos. None if the log has nothing yet."""
    try:
        with log_path.open("rb") as handle:
            handle.seek(start_pos)
            chunk = handle.read(65536).decode("utf-8", "replace")
    except OSError:
        return None
    matches = PROGRESS_RE.findall(chunk.replace("\r", "\n"))
    if not matches:
        return None
    step, total, eta = matches[-1]
    return int(step), int(total), eta


def render_bar(step: int, total: int, eta: str, width: int = 28) -> str:
    filled = int(width * step / max(total, 1))
    return f"\r  [{'█' * filled}{'·' * (width - filled)}] {step}/{total}  eta {eta}   "


def _images_from_outputs(outputs: dict) -> list[dict]:
    for node_output in outputs.values():
        if isinstance(node_output, dict) and node_output.get("images"):
            return [i for i in node_output["images"] if isinstance(i, dict) and i.get("filename")]
    return []


def generate_local(prompt: str, output: Path, width: int = 1024, height: int = 1024,
                   steps: int = 20, checkpoint: str | None = None, seed: int | None = None,
                   negative_prompt: str = "", url: str = DEFAULT_COMFY_URL,
                   timeout: int = 600, poll: float = 2.0) -> Path:
    ok, models, error = check_comfy(url)
    if not ok:
        raise ShadeForgeError(
            "Local ComfyUI is unavailable. Start ComfyUI, or pass --provider gemini to use "
            f"cloud generation. Detail: {error}"
        )
    if not models:
        raise ShadeForgeError(
            "ComfyUI is running but exposes no checkpoints. Install an SD/SDXL checkpoint in "
            "ComfyUI/models/checkpoints and refresh ComfyUI."
        )
    selected = checkpoint or models[0]
    if selected not in models:
        raise ShadeForgeError(
            f"Checkpoint '{selected}' not found. Available: {', '.join(models[:12])}"
        )
    seed = random.randint(1, 2**31 - 1) if seed is None else seed
    workflow = build_comfy_workflow(prompt, selected, width, height, steps, seed, negative_prompt)
    payload = {"prompt": workflow, "client_id": f"shadeforge-{random.randint(10**5, 10**6 - 1)}"}
    result = request_json(f"{url}/prompt", method="POST", data=payload, timeout=30)
    if result.get("node_errors"):
        raise ShadeForgeError(
            "ComfyUI rejected the workflow (likely an unsupported model or missing node): "
            + json.dumps(result["node_errors"])[:800]
        )
    prompt_id = result.get("prompt_id")
    if not prompt_id:
        raise ShadeForgeError(f"ComfyUI did not queue the workflow: {json.dumps(result)[:800]}")
    print(f"Queued local generation (seed {seed}, model {selected}).")

    # Live step counter, read from the server log ShadeForge itself started.
    log_path = setup_mod.COMFY_DIR / "comfyui.log"
    log_start = log_path.stat().st_size if log_path.exists() else 0
    show_bar = sys.stdout.isatty() and log_path.exists()
    last_line = ""
    write_progress_state(log_path, log_start)

    deadline = time.monotonic() + timeout
    while True:
        if show_bar:
            progress = read_comfy_progress(log_path, log_start)
            if progress:
                last_line = render_bar(*progress)
                sys.stdout.write(last_line)
                sys.stdout.flush()
        try:
            history = request_json(f"{url}/history/{urllib.parse.quote(str(prompt_id))}", timeout=10)
        except ShadeForgeError:
            history = {}
        record = history.get(prompt_id, {}) if isinstance(history, dict) else {}
        images = _images_from_outputs(record.get("outputs", {}) or {})
        if images:
            if last_line:
                sys.stdout.write("\n")
            _download_comfy_image(url, images[0], output)
            return output
        status = record.get("status", {}) or {}
        if status.get("status_str") == "error":
            raise ShadeForgeError("ComfyUI reported an execution error. Check its console log.")
        if status.get("completed"):
            raise ShadeForgeError("ComfyUI finished the job but produced no image output.")
        if time.monotonic() >= deadline:
            raise ShadeForgeError(
                f"Timed out after {timeout}s waiting for ComfyUI. The job may still be running."
            )
        time.sleep(poll)


def ratio_label(width: int, height: int) -> str:
    ratio = width / height
    known = [(1, 1), (16, 9), (9, 16), (4, 3), (3, 4), (3, 2), (2, 3), (5, 4), (4, 5), (21, 9)]
    w, h = min(known, key=lambda item: abs((item[0] / item[1]) - ratio))
    return f"{w}:{h}"


def _image_from_candidates(result: dict) -> str | None:
    for candidate in result.get("candidates") or []:
        if not isinstance(candidate, dict):
            continue
        for part in (candidate.get("content") or {}).get("parts") or []:
            blob = part.get("inlineData") or part.get("inline_data") if isinstance(part, dict) else None
            if isinstance(blob, dict) and blob.get("data"):
                return blob["data"]
    return None


def generate_gemini(prompt: str, output: Path, width: int = 1024, height: int = 1024,
                    model: str = DEFAULT_GEMINI_MODEL) -> Path:
    key = get_gemini_key()
    if not key:
        raise ShadeForgeError("No Gemini API key detected. Set GEMINI_API_KEY in your environment.")
    endpoint = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{urllib.parse.quote(model)}:generateContent"
    )
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseModalities": ["IMAGE"],
            "imageConfig": {
                "aspectRatio": ratio_label(width, height),
                "imageSize": "1K" if max(width, height) <= 1024 else "2K",
            },
        },
    }
    try:
        result = request_json(endpoint, method="POST", data=body,
                              headers={"x-goog-api-key": key}, timeout=300)
    except ShadeForgeError as exc:
        # "limit: 0" is not an exhausted daily quota -- it means the tier grants no image
        # requests at all. Free-tier Gemini keys cannot generate images, period.
        if "429" in str(exc) and "limit: 0" in str(exc):
            raise ShadeForgeError(
                f"Gemini image generation is not available on this API key's tier "
                f"(quota limit is 0 for {model}, not merely exhausted). Image models require "
                "billing enabled on the key's Google Cloud project -- the free tier covers text "
                "models only. Enable billing at https://aistudio.google.com/apikey, or use "
                "--provider local."
            ) from exc
        raise
    image_b64 = _image_from_candidates(result if isinstance(result, dict) else {})
    if not image_b64:
        blocked = (result.get("promptFeedback") or {}).get("blockReason") if isinstance(result, dict) else None
        raise ShadeForgeError(
            f"Gemini returned no image (blockReason={blocked})." if blocked else
            "Gemini returned no image. Check model name, access and quota."
        )
    try:
        image_bytes = base64.b64decode("".join(image_b64.split()), validate=True)
    except (ValueError, TypeError) as exc:
        raise ShadeForgeError("Gemini returned image data that could not be decoded.") from exc
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(image_bytes)
    return output


def grayscale_preview(path: Path, columns: int = 64) -> None:
    # True-color ANSI blocks are a picture in a terminal and thousands of junk
    # escape codes anywhere else (pipes, CI, an agent's tool output). Only draw
    # when something can actually render it.
    if not sys.stdout.isatty():
        print("(preview skipped: output is not a terminal -- open the file to view it)")
        return
    try:
        from PIL import Image
    except ImportError:
        print("Tip: install Pillow for the terminal preview: python3 -m pip install Pillow")
        return
    try:
        img = Image.open(path).convert("L")
        rows = max(8, int(columns * (img.height / max(img.width, 1)) / 2))
        img = img.resize((columns, rows * 2))
        pixels = list(img.getdata())
        print("\n\033[90m--- SHADEFORGE / MONO PREVIEW ---\033[0m")
        for y in range(0, rows * 2, 2):
            line = []
            for x in range(columns):
                top = pixels[y * columns + x]
                bottom = pixels[(y + 1) * columns + x]
                line.append(f"\033[38;2;{top};{top};{top}m\033[48;2;{bottom};{bottom};{bottom}m▀")
            print("".join(line) + "\033[0m")
        print("\033[90m--- END PREVIEW ---\033[0m\n")
    except Exception as exc:
        print(f"Preview unavailable: {exc}")


def open_image(path: Path) -> bool:
    try:
        if sys.platform == "darwin":
            subprocess.Popen(["open", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif sys.platform.startswith("win"):
            os.startfile(str(path))  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except OSError:
        return False


def default_output() -> Path:
    return Path("generated") / f"shadeforge-{datetime.now():%Y%m%d-%H%M%S}.png"


def best_checkpoint(models: list[str]) -> str | None:
    """Pick the most capable installed checkpoint, not just the first alphabetically."""
    if not models:
        return None
    def rank(name: str) -> int:
        low = name.lower()
        for i, pref in enumerate(setup_mod.CHECKPOINT_PREFERENCE):
            if pref in low:
                return i
        return len(setup_mod.CHECKPOINT_PREFERENCE)
    return min(models, key=rank)


def warn_oversized(width: int, height: int, checkpoint: str | None) -> None:
    name = (checkpoint or "").lower()
    looks_sd15 = "v1-5" in name or "sd15" in name or "sd-v1" in name
    if looks_sd15 and max(width, height) > setup_mod.SD15_MAX_EDGE:
        print(f"Note: {width}x{height} exceeds SD1.5's comfort zone "
              f"({setup_mod.SD15_MAX_EDGE}px max edge). Expect duplicated subjects. "
              "Use a smaller size, or an SDXL checkpoint for large canvases.")


def cmd_generate(args: argparse.Namespace) -> int:
    output = Path(args.output or default_output()).expanduser().resolve()
    provider = args.provider
    key_present = bool(get_gemini_key())

    if provider in ("auto", "local") and args.auto_setup:
        # Install and/or start the local backend rather than failing with instructions.
        # Don't surprise-download 4 GB when an explicit cloud fallback was already offered.
        allow_install = provider == "local" or not (args.allow_cloud and key_present)
        try:
            setup_mod.ensure_local(args.comfy_url, auto=True, allow_install=allow_install)
        except setup_mod.SetupError as exc:
            if provider == "local":
                raise ShadeForgeError(str(exc)) from exc
            print(f"Local backend unavailable: {exc}")

    if provider == "auto":
        local_ok, models, local_err = check_comfy(args.comfy_url)
        if local_ok and models:
            provider = "local"
        elif args.allow_cloud and key_present:
            provider = "gemini"
            print("Local backend unavailable; using Gemini because --allow-cloud was set.")
        else:
            if key_present:
                print("Gemini key detected, but cloud use is opt-in: "
                      "re-run with --provider gemini or --allow-cloud.")
            raise ShadeForgeError(
                "No usable local model. Run `shadeforge setup`, or enable cloud "
                f"generation explicitly. Detail: {local_err or 'no checkpoints installed'}"
            )

    if provider == "local":
        installed = check_comfy(args.comfy_url)[1]
        if not args.checkpoint:
            args.checkpoint = best_checkpoint(installed)
        warn_oversized(args.width, args.height, args.checkpoint)
        if typeset.wants_text(args.prompt) and not any(
                k in (args.checkpoint or "").lower() for k in ("sdxl", "flux")):
            print(
                "Guardrail: this prompt asks for words in the image, and no local SD1.5-class "
                "model can spell -- it will render label-shaped gibberish.\n"
                "  Remove the text from the prompt, then set it with --headline/--sub/--cta, "
                "which composites real type at full resolution."
            )
        path = generate_local(args.prompt, output, args.width, args.height, args.steps,
                              args.checkpoint, args.seed, args.negative_prompt,
                              args.comfy_url, args.timeout)
    else:
        print(f"Using Gemini cloud generation ({args.gemini_model}). Prompt is sent to Google.")
        path = generate_gemini(args.prompt, output, args.width, args.height, args.gemini_model)

    if any([args.headline, args.kicker, args.sub, args.benefit, args.cta]):
        if not typeset.available():
            print("Typesetting unavailable (needs Pillow and system fonts); skipping.")
        else:
            typed = path.with_name(path.stem + "-typeset.png")
            ig = (1080, 1350) if args.height > args.width else (1080, 1080)
            typeset.build_ad(path, typed, headline=args.headline, kicker=args.kicker,
                             sub=args.sub, benefit=args.benefit, cta=args.cta,
                             size=ig if args.instagram else None)
            print(f"Typeset layout written: {typed}")
            path = typed

    print(f"\nIMAGE_PATH={path}")
    print(f"Provider: {provider}")
    if args.preview:
        grayscale_preview(path, args.preview_width)
    if args.open:
        print("Opened image in the system viewer." if open_image(path)
              else "Could not open image automatically.")
    print(f"Markdown: ![Generated image]({path})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="shadeforge",
                                     description="Local-first image generation for Claude Code.")
    parser.add_argument("--version", action="version", version=f"ShadeForge {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_doc = sub.add_parser("doctor", help="Check ComfyUI, checkpoints and Gemini key presence.")
    p_doc.add_argument("--comfy-url", default=DEFAULT_COMFY_URL)
    p_doc.set_defaults(func=lambda a: doctor(a.comfy_url))

    p_gen = sub.add_parser("generate", help="Generate an image locally, or explicitly with Gemini.")
    p_gen.add_argument("prompt", help="Image description")
    p_gen.add_argument("--provider", choices=["auto", "local", "gemini"], default="auto")
    p_gen.add_argument("--allow-cloud", action="store_true",
                       help="Let auto mode fall back to Gemini when local is unavailable.")
    p_gen.add_argument("--comfy-url", default=DEFAULT_COMFY_URL)
    p_gen.add_argument("--checkpoint", help="Exact ComfyUI checkpoint filename")
    p_gen.add_argument("--gemini-model", default=DEFAULT_GEMINI_MODEL)
    p_gen.add_argument("--output", help="Output path (default: generated/ with timestamp)")
    p_gen.add_argument("--width", type=int, default=1024)
    p_gen.add_argument("--height", type=int, default=1024)
    p_gen.add_argument("--steps", type=int, default=20)
    p_gen.add_argument("--seed", type=int)
    p_gen.add_argument("--negative-prompt", default="low quality, blurry, distorted, artifacts")
    p_gen.add_argument("--timeout", type=int, default=1800)
    p_gen.add_argument("--preview", action=argparse.BooleanOptionalAction, default=True,
                       help="ANSI grayscale preview in the terminal (default: on, TTY only).")
    p_gen.add_argument("--preview-width", type=int, default=64,
                       help="Preview width in columns (default: 64).")
    p_gen.add_argument("--open", action=argparse.BooleanOptionalAction, default=True,
                       help="Open the image in the system viewer (default: on).")
    p_gen.add_argument("--auto-setup", action=argparse.BooleanOptionalAction, default=True,
                       help="Install and start the local backend automatically (default: on).")
    for name, help_text in [("headline", "Main headline, set in a serif face"),
                            ("kicker", "Small brand line above the headline"),
                            ("sub", "Supporting line under the headline"),
                            ("benefit", "Benefit line near the bottom"),
                            ("cta", "Call to action, drawn in a bordered box")]:
        p_gen.add_argument(f"--{name}", default="", help=f"{help_text} (real type, composited)")
    p_gen.add_argument("--instagram", action="store_true",
                       help="Upscale the typeset layout to 1080x1350 (or 1080x1080).")
    p_gen.set_defaults(func=cmd_generate)

    p_setup = sub.add_parser("setup", help="Install ComfyUI, a pinned torch stack and a model.")
    p_setup.add_argument("--skip-model", action="store_true", help="Install code only, no model.")
    p_setup.add_argument("--model", default="auto",
                         choices=["auto", *setup_mod.MODELS],
                         help="Checkpoint to install; 'auto' picks the best that fits the disk.")
    p_setup.set_defaults(func=lambda a: (setup_mod.install(a.skip_model, a.model), 0)[1])

    p_disk = sub.add_parser("models", help="Show disk space and which checkpoints fit.")
    p_disk.set_defaults(func=lambda a: (setup_mod.report_disk(setup_mod.pick_model()), 0)[1])

    p_type = sub.add_parser("typeset", help="Composite real typography onto an existing image.")
    p_type.add_argument("image")
    p_type.add_argument("output")
    for name in ("headline", "kicker", "sub", "benefit", "cta"):
        p_type.add_argument(f"--{name}", default="")
    p_type.add_argument("--instagram", action="store_true")
    p_type.set_defaults(func=lambda a: (typeset.build_ad(
        Path(a.image), Path(a.output), headline=a.headline, kicker=a.kicker, sub=a.sub,
        benefit=a.benefit, cta=a.cta,
        size=(1080, 1350) if a.instagram else None), 0)[1])

    p_start = sub.add_parser("start", help="Start the local ComfyUI server in the background.")
    p_start.add_argument("--comfy-url", default=DEFAULT_COMFY_URL)
    p_start.set_defaults(func=lambda a: (setup_mod.start_server(a.comfy_url), 0)[1])
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "generate":
        for name in ("width", "height"):
            value = getattr(args, name)
            if not 256 <= value <= 2048:
                parser.error(f"{name} must be between 256 and 2048")
            if value % 8:
                parser.error(f"{name} must be divisible by 8")
        if not 1 <= args.steps <= 60:
            parser.error("steps must be between 1 and 60")
    try:
        return int(args.func(args))
    except setup_mod.SetupError as exc:
        print(f"ShadeForge setup error: {exc}", file=sys.stderr)
        return 3
    except ShadeForgeError as exc:
        print(f"ShadeForge error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        return 130
    finally:
        # Covers success, failure and Ctrl-C, so no stale bar is left behind.
        clear_progress_state()


if __name__ == "__main__":
    raise SystemExit(main())
