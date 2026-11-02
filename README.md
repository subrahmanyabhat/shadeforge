# ShadeForge

**Local-first image generation for Claude Code.** Prompts go to a local ComfyUI instance by default. Cloud (Gemini) generation exists but is strictly opt-in: a detected API key never causes an upload on its own.

v0.4.0 — 50 tests against an in-process fake ComfyUI server. **Exercised in anger on an Intel Mac only.** The Apple Silicon and Linux code paths are written but have never been run; treat them as untested and report what breaks.

## One command

```bash
shadeforge generate "a cinematic lighthouse in sea mist"
```

First run installs ComfyUI and a model (~4 GB, once), starts the server, generates with a live step bar, prints a grayscale preview, and opens the image. Later runs skip straight to generating.

## Commands

- `shadeforge generate "..."` — install if needed, start if stopped, generate. `--preview`, `--open` and `--auto-setup` default **on** (`--no-*` to disable).
- `shadeforge typeset in.png out.png --headline "..."` — composite real typography onto an image.
- `shadeforge models` — free disk and which checkpoints fit.
- `shadeforge setup` — install explicitly. `--model realistic-vision|sd15|auto`, `--skip-model`.
- `shadeforge start` — start the server in the background (log: `~/ComfyUI/comfyui.log`).
- `shadeforge doctor` — reachability, checkpoints, Gemini key presence (never the value). Exit 0 = ready.
- `--provider gemini` — explicit cloud generation. See the quota note below.
- Claude Code plugin skill + status line in `claude-plugin/`.

## Text in images

**No local model can spell.** SD1.5/SDXL checkpoints render label-shaped gibberish. ShadeForge warns when a prompt asks for words, and provides the path that actually works: generate the scene with no text, then composite real system fonts over it at full resolution.

```bash
shadeforge generate "RAW photo, amber serum bottle on black marble, gold rim light, 85mm" \
  --width 512 --height 640 --steps 28 \
  --kicker "MAISON LUXE" --headline "LUXE HAIR SERUM" \
  --sub "SMOOTH • NOURISH • SHINE" --cta "DISCOVER THE RITUAL" --instagram
```

Writes `<name>-typeset.png` at 1080×1350. Headlines auto-shrink to fit rather than clipping, and contrast scrims are drawn only behind the text that exists.

## Model selection

`setup` reports free disk and installs the best checkpoint that fits, refusing rather than filling the disk. `generate` auto-selects the most capable installed checkpoint (photoreal fine-tunes outrank SD1.5 base).

| Model | Size | Use for |
| --- | --- | --- |
| realistic-vision (default) | 2.0 GB | Products, people, food, interiors — far better adherence |
| sd15 | 4.0 GB | Generic scenes only; fallback |

## Progress display

A live step bar renders in the Claude Code status line **of the session that started the generation only** — ShadeForge records the owning session in `~/.shadeforge/progress.json` and the status line checks it, so one render doesn't paint bars across every open terminal. Install `claude-plugin/scripts/statusline.sh`; it chains to any statusline you already have via `SHADEFORGE_STATUSLINE_BASE`.

The ANSI `--preview` only draws on a real TTY and self-skips when piped.

## Requirements

- Python 3.10+
- Local generation: ComfyUI running at `http://127.0.0.1:8188` with at least one checkpoint in `ComfyUI/models/checkpoints`. No model is bundled; check its licence yourself.
- Preview: Pillow.
- Cloud: `GEMINI_API_KEY` or `GOOGLE_API_KEY` in the environment.

## Install

```bash
git clone https://github.com/subrahmanyabhat/shadeforge
cd shadeforge
python3 -m pip install -e '.[preview]'
shadeforge doctor
```

If pip refuses because of a managed Python:

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -e '.[preview]'
```

`shadeforge` only exists in the environment you installed into. Without installing, `python3 -m shadeforge` works from the repo root.

## Claude Code plugin

```bash
claude --plugin-dir $(pwd)/claude-plugin
```

Skill name: `shadeforge-image`. The skill calls the `shadeforge` command, and falls back to `python3 "${CLAUDE_PLUGIN_ROOT}/scripts/shadeforge.py"` when the command isn't on PATH — so the plugin works even without `pip install`.

## Platform notes (why the pins exist)

**Intel Mac (x86_64 darwin).** PyTorch stopped shipping macOS x86_64 wheels after **2.2.2**, so `setup` pins torch 2.2.2 + torchvision 0.17.2 and checks out a ComfyUI commit from before 2024-04-01 to match. It also forces `numpy<2` — torch 2.2.2 was built against numpy 1.x and breaks on the array bridge otherwise. Needs Python 3.11 (3.12+ has no 2.2.2 wheel); `setup` looks in PATH, uv (`~/.local/bin`), Homebrew and pyenv. No Metal/MPS on Intel, so generation is CPU-only: **~15 s/step**, about 5–6 min for 20 steps at 768×432.

**Apple Silicon / Linux** get current ComfyUI and current torch — no pinning.

## Image size vs checkpoint

SD1.5 is trained at 512px and **duplicates subjects past ~768px on an edge**. For a 16:9 banner use 768×432, not 1024×576. ShadeForge prints a warning when the requested size exceeds what the loaded checkpoint handles. Use an SDXL checkpoint for larger canvases.

## Prompt adherence

Three traps, each hit for real:

**Never name the artifact type.** `banner`, `poster`, `billboard`, `flyer`, `mockup` make the model draw that physical object — ask for a "coconut banner" and you get a billboard on a stand with fake lettering. Describe what is *in* the picture; the aspect ratio carries the format.

**It substitutes the common form of a noun.** Ask for a *tender* (green, unhusked) coconut and you get a mature brown one. Counter it in the **positive** prompt by front-loading and repeating the distinguishing adjectives: `whole green unhusked young coconut, smooth green skin`.

**Negatives are for style and defects only:**

```bash
--negative-prompt "low quality, blurry, text, watermark, deformed, duplicate"
```

Do *not* negate the subject. Negating `brown coconut, cracked coconut` deleted every coconut from the frame rather than correcting their type — a negative removes the whole concept, it does not steer it.

## Gemini quota reality

**Free-tier keys cannot generate images at all.** Every image model returns HTTP 429 with `limit: 0` — that is "this tier grants zero image requests", not "you used today's allowance". Billing must be enabled on the key's Cloud project. ShadeForge detects this specific case and says so instead of reporting a generic rate limit.

## ComfyUI setup (manual, if you skip `setup`)

1. Install and start ComfyUI (https://docs.comfy.org/).
2. Drop an SD/SDXL checkpoint into `ComfyUI/models/checkpoints`.
3. `shadeforge doctor`.

Checkpoints are discovered from `/object_info/CheckpointLoaderSimple` (with `/models/checkpoints` as a fallback). The workflow uses the stock nodes `CheckpointLoaderSimple`, `CLIPTextEncode`, `EmptyLatentImage`, `KSampler`, `VAEDecode`, `SaveImage`. Flux/SD3/custom architectures need a different graph — ShadeForge will surface ComfyUI's `node_errors` rather than hanging.

Older Intel Macs: start at 512×512, 8–12 steps, watch memory.

## Usage

```bash
shadeforge doctor

# local (auto mode prefers local, never touches the cloud on its own)
shadeforge generate "cinematic black-and-white lighthouse in sea mist" \
  --width 512 --height 512 --steps 12 --preview --open

# pick a checkpoint and a banner ratio
shadeforge generate "coastal garden restaurant banner, warm daylight" \
  --provider local --checkpoint sdxl.safetensors --width 1024 --height 576

# explicit cloud — this sends your prompt to Google
export GEMINI_API_KEY=...
shadeforge generate "paper-cut blue coastal house" --provider gemini

# allow cloud only as a fallback when local is unavailable
shadeforge generate "coastal restaurant concept" --allow-cloud
```

Default Gemini model is `gemini-3-pro-image-preview`; override with `--gemini-model` or `GEMINI_IMAGE_MODEL`. ComfyUI URL overrides with `--comfy-url` or `COMFYUI_URL`.

## Privacy and cost

- `auto` picks local only when ComfyUI is reachable *and* has a checkpoint. Otherwise it stops with an error — even if a key is sitting in the environment.
- `--provider gemini` / `--allow-cloud` are the only paths that send a prompt off-machine, and both are explicit.
- The key is read from env vars only. ShadeForge does not read `.env` files, shell history, or keychains, and never prints the value.
- Local generation costs no API fees but does cost CPU/GPU, RAM, power and disk.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| "Local ComfyUI is unavailable" | Start ComfyUI; check `COMFYUI_URL`. |
| "exposes no checkpoints" | Install a checkpoint, then refresh ComfyUI. |
| "rejected the workflow" | Model architecture needs a different graph; see the ComfyUI console. |
| "Timed out after 600s" | Big image on slow hardware — drop to 512×512 / fewer steps, or raise `--timeout`. |
| "Gemini returned no image" | Check model name, access and quota. `blockReason` in the message means the prompt was filtered. |

## Tests

```bash
cd ~/Projects/shadeforge && python3 -m unittest discover -s tests -t . -v
```

Tests spin up an in-process fake ComfyUI server — no GPU, no network, no API key.

## License

MIT. ComfyUI, models and the Gemini API carry their own terms.
