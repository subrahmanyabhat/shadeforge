---
name: shadeforge-image
description: Generate, typeset and revise image assets locally via ComfyUI — installs and starts the backend on first use, picks a model that fits the disk, and composites real typography for any image that needs words. Optional explicit Gemini cloud path. Use when the user asks to create, generate, draw, design, or revise an image, banner, ad, poster, or product shot.
---

# ShadeForge image generation

Local-first. One command installs, starts, generates. Nothing is sent to any cloud unless the user explicitly asks.

## Decide first: does the image need words?

This single question routes everything, so answer it before generating.

| The image needs… | Do this |
| --- | --- |
| **No words** (photo, texture, scene, product shot) | Generate directly. |
| **Words** (ad, banner, poster, label, logo) | Generate the scene **with no text in the prompt**, then set type with `--headline` / `--kicker` / `--sub` / `--benefit` / `--cta`. |

**No local model can spell.** SD1.5- and SDXL-class checkpoints render label-shaped gibberish — "GHEN KEHSR" where you asked for a brand name. This is a hard architectural limit, not a prompt problem, and no amount of retrying fixes it. ShadeForge detects text-seeking prompts and warns, but you should route correctly without being told.

The typeset path produces genuinely crisp, correctly spelled type at full resolution, because it composites real system fonts over the render — the same way beauty and fashion campaigns are actually produced.

## Let it choose the model

Do not pass `--checkpoint` unless the user named one. ShadeForge reads the prompt and routes:
photographic cues to Realistic Vision, illustration cues to DreamShaper, oversized canvases to
SDXL, and text-bearing prompts to the composited-typography path. It prints its reasoning; relay
that reasoning to the user rather than presenting the choice as arbitrary.

`shadeforge recommend "<prompt>"` shows the decision without rendering — useful when the user is
deciding what to ask for, and free.

## The commands

```bash
# scene only
shadeforge generate "<visual prompt>" --width 768 --height 432

# full ad: scene + real typography, Instagram-sized
shadeforge generate "<visual prompt, no words in it>" \
  --width 512 --height 640 --steps 28 \
  --kicker "MAISON LUXE" --headline "LUXE HAIR SERUM" \
  --sub "SMOOTH • NOURISH • SHINE" \
  --benefit "WEIGHTLESS SHINE, VISIBLY SMOOTHER HAIR" \
  --cta "DISCOVER THE RITUAL" --instagram

# typeset an image that already exists
shadeforge typeset in.png out.png --headline "..." --cta "..." --instagram

shadeforge doctor    # ready? exit 0 = yes
shadeforge models    # disk free + which checkpoints fit
shadeforge setup     # explicit install; --model realistic-vision|sd15, --skip-model
shadeforge start     # start the server only
```

Not on PATH? Use `python3 "${CLAUDE_PLUGIN_ROOT}/scripts/shadeforge.py"` — it works without installing.

`--preview`, `--open` and `--auto-setup` default **on**. The preview is a true-colour ANSI render that only appears on a real terminal; it self-skips when output is piped (where it would just be thousands of escape codes). You should still Read the file to actually see the result.

## First run

`generate` installs and starts everything automatically. Setup reports free disk and picks the best checkpoint that fits, refusing rather than filling the disk:

- **realistic-vision** (2 GB) — the default. Photoreal, far better prompt adherence. Use it for products, people, food, interiors.
- **sd15** (4 GB) — base model, generic scenes only. Only if Realistic Vision is unavailable.

Setup needs the model size plus ~3 GB headroom. If it refuses, say what's short and suggest the usual wins (Xcode `iOS DeviceSupport`, Android `system-images`) rather than deleting anything yourself.

## Sizing

SD1.5-class models are trained at 512px and **duplicate subjects past ~768px on any edge** — two bottles, three hands. Generate small, upscale in the typeset step.

| Target | Generate at | Then |
| --- | --- | --- |
| Instagram portrait 4:5 | 512×640 | `--instagram` → 1080×1350 |
| Instagram square | 512×512 | `--instagram` → 1080×1080 |
| Wide banner 16:9 | 768×432 | — |

## Prompting that works

- **Never name the artifact type.** `banner`, `poster`, `billboard`, `flyer`, `mockup` make the model draw that physical object — a billboard on a stand, not your scene. Describe only what is *in* the picture; the aspect ratio carries the format.
- **Lead with `RAW photo,`** and close with camera language (`85mm`, `shallow depth of field`, `studio softbox`) for photoreal work.
- **Repeat distinguishing adjectives.** The model substitutes the common form of a noun — ask for a *tender* (green, unhusked) coconut and you get a mature brown one unless you insist: `whole green unhusked young coconut, smooth green skin`.
- **Negatives are for style and defects only:** `low quality, blurry, text, watermark, deformed, duplicate`. Never negate the subject — negating `brown coconut` removed every coconut from the frame instead of correcting its type.

## Speed

Intel Mac is CPU-only (no Metal/MPS). **~15 s/step at SD1.5, ~22 s/step at Realistic Vision.** A 28-step 512×640 render takes ~10 minutes. Say this up front. Use `--steps 10-12` while iterating on composition, then a final pass at 25-30. Apple Silicon is roughly 15× faster.

A live progress bar appears in the status line of the session that started the run, and only that session.

## Gemini

Only with explicit per-generation permission. Note that **free-tier keys cannot generate images at all** — every image model returns quota `limit: 0` until billing is enabled on the key's Cloud project. ShadeForge reports this precisely; don't relay it as a temporary rate limit.

## Guardrails

- **Look before reporting.** Read the `IMAGE_PATH` with the Read tool and judge it against the brief. Never describe an image you have not seen.
- **Never claim success without evidence** — exit code 0 *and* the file exists. A failed run is a failed run; say so.
- **Name what is missing.** If the brief asked for flowing hair and the render has none, lead with that. Do not list what went right and bury the miss.
- **Stop after two failed attempts** at the same subject. Each local run costs 5–10 minutes. Report the gap and offer a better checkpoint or the cloud path instead of burning a third.
- **Never put a real brand's name, logo, or trade dress** into a generated image unless the user owns the brand. Don't generate images of real identifiable people.
- **Composited labels are composites.** When type is pasted onto a generated bottle or pack, say so — it is flat against a curved surface and will not survive print scrutiny.
- **Don't delete anything to free space.** Report what's short and let the user choose.
- **Never print API key values**, and never read `.env` files, shell history or keychains looking for one.
- **State the cost model once**: local is free but slow and uses the laptop's CPU, RAM and power; Gemini is fast but billed and sends the prompt to Google.
