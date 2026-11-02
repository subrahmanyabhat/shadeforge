"""Pick the model from the prompt, and explain the choice.

ComfyUI makes you choose a checkpoint and wire a graph before you can draw
anything. The checkpoint that makes a good product photo is the wrong one for a
flat illustration, and none of the small ones can spell. That knowledge usually
lives in the user's head; here it lives in code, and the reasoning is printed so
the choice can be argued with rather than guessed at.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .typeset import wants_text

# Style cues. Deliberately narrow: a wrong guess here sends the render to the
# wrong model, so only match words that genuinely signal a medium.
ART_CUES = re.compile(
    r"(?i)\b(illustration|illustrated|painting|painted|watercolou?r|gouache|oil\s+painting|"
    r"anime|manga|cartoon|comic|sketch|drawing|line\s*art|vector|flat\s+design|"
    r"concept\s+art|storybook|fantasy\s+art|pixel\s+art|woodcut|linocut|collage)\b")

PHOTO_CUES = re.compile(
    r"(?i)\b(photo|photograph|photorealistic|realistic|raw\s+photo|product\s+shot|"
    r"packshot|portrait|headshot|studio|cinematic|bokeh|dslr|\d{2,3}\s*mm|f/\d|"
    r"macro|editorial|lookbook|food\s+photography)\b")

MARK_CUES = re.compile(r"(?i)\b(logo|logotype|wordmark|emblem|monogram|icon\s+set|favicon)\b")


@dataclass
class Intent:
    text: bool = False
    style: str = "general"       # photo | art | general
    mark: bool = False           # a logo or icon, which raster diffusion handles badly
    big_canvas: bool = False


@dataclass
class Decision:
    model: str | None                 # checkpoint filename, or None if nothing usable
    intent: Intent
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    typeset: bool = False             # compose real type after rendering
    missing: str | None = None        # catalogue key worth installing for this prompt


def classify(prompt: str, width: int = 512, height: int = 512) -> Intent:
    art, photo = bool(ART_CUES.search(prompt)), bool(PHOTO_CUES.search(prompt))
    return Intent(
        text=wants_text(prompt),
        # When a prompt claims both, photo wins: "photorealistic illustration" is
        # overwhelmingly used to mean a rendered-looking photo.
        style="photo" if photo else ("art" if art else "general"),
        mark=bool(MARK_CUES.search(prompt)),
        big_canvas=max(width, height) > 768,
    )


def _has(installed: list[str], *needles: str) -> str | None:
    for name in installed:
        low = name.lower()
        if any(n in low for n in needles):
            return name
    return None


def decide(prompt: str, installed: list[str], width: int = 512, height: int = 512,
           legacy_mac: bool = False) -> Decision:
    """Choose a checkpoint for this prompt from what is actually installed."""
    intent = classify(prompt, width, height)
    d = Decision(model=None, intent=intent)

    flux = _has(installed, "flux")
    sdxl = _has(installed, "sdxl", "sd_xl")
    photo = _has(installed, "realisticvision", "realistic_vision", "epicrealism")
    art = _has(installed, "dreamshaper", "anything", "rev-animated")
    base = _has(installed, "v1-5", "sd15", "v1.5")

    # 1. Words in the image. Only Flux-class models can spell; everything else
    #    gets the composited-typography path instead of a doomed attempt.
    if intent.text:
        if flux and not legacy_mac:
            d.model, d.typeset = flux, False
            d.reasons.append("Prompt asks for words, and Flux can actually render legible text.")
        else:
            d.typeset = True
            if flux and legacy_mac:
                d.warnings.append(
                    "Flux is installed but unusable here: this Intel Mac runs a 2024-era ComfyUI "
                    "with no Flux support, on CPU. Rendering would take hours.")
            else:
                d.missing = "flux-schnell"
            d.reasons.append(
                "Prompt asks for words. No local model here can spell, so the scene is rendered "
                "without text and real type is composited on top.")

    # 2. Medium decides the checkpoint when text has not already settled it.
    if d.model is None:
        if intent.style == "art" and art:
            d.model = art
            d.reasons.append("Illustration cues in the prompt, so a stylised checkpoint fits better.")
        elif intent.style == "art" and not art:
            d.missing = d.missing or "dreamshaper"
            d.reasons.append("Illustration asked for, but only photoreal checkpoints are installed.")
        elif intent.big_canvas and sdxl:
            d.model = sdxl
            d.reasons.append(f"{width}x{height} is past SD1.5's limit, and SDXL is built for it.")
        elif intent.style == "photo" and photo:
            d.model = photo
            d.reasons.append("Photographic cues, so the photoreal fine-tune wins.")

    # 3. Anything still unresolved falls back by general capability.
    if d.model is None:
        d.model = photo or sdxl or art or base or (installed[0] if installed else None)
        if d.model:
            # Don't claim there was no signal when a signal was found but unserviceable --
            # the earlier reason already explained what was missing.
            d.reasons.append(
                f"Falling back to the most capable installed checkpoint ({d.model})."
                if d.reasons else
                f"No medium cues in the prompt; using the most capable installed "
                f"checkpoint ({d.model}).")

    if d.model is None:
        d.reasons.append("No checkpoints installed. Run `shadeforge setup`.")
        return d

    # Cross-checks that do not change the model but change the outcome.
    if intent.big_canvas and not _has([d.model], "sdxl", "sd_xl", "flux"):
        d.warnings.append(
            f"{width}x{height} exceeds what {d.model} handles well; expect duplicated subjects. "
            "Render smaller, or install SDXL.")
    if intent.mark:
        d.warnings.append(
            "Logos and icons want vectors, not diffusion. Use this for a texture or a mood board, "
            "then redraw the mark properly.")
    return d


def explain(d: Decision) -> str:
    """Human-readable routing summary, printed before a render starts."""
    out = []
    if d.model:
        out.append(f"Model: {d.model}" + ("  + composited typography" if d.typeset else ""))
    for r in d.reasons:
        out.append(f"  why: {r}")
    for w in d.warnings:
        out.append(f"  note: {w}")
    if d.missing:
        out.append(f"  tip: `shadeforge setup --model {d.missing}` would suit this prompt better.")
    return "\n".join(out)
