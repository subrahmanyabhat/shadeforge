"""Composite real typography onto a generated scene.

No diffusion model in the SD1.5/SDXL family can spell. Asking for text in the
prompt yields label-shaped gibberish. Campaigns are produced by rendering the
scene and setting type on top at full resolution -- which is what this does.
"""
from __future__ import annotations

import re
from pathlib import Path

GOLD = (212, 175, 110)
IVORY = (245, 240, 232)
INK = (46, 33, 18)

# System faces, best-to-worst for a luxury register. First hit wins.
# macOS first, then Linux distro fonts, then Windows -- the feature must not be
# macOS-only just because that is where it was written.
SERIF_FACES = [
    "/System/Library/Fonts/Supplemental/Didot.ttc",
    "/System/Library/Fonts/Supplemental/Bodoni 72.ttc",
    "/System/Library/Fonts/Supplemental/Baskerville.ttc",
    "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    "/usr/share/fonts/TTF/DejaVuSerif.ttf",
    "C:/Windows/Fonts/georgia.ttf",
    "C:/Windows/Fonts/times.ttf",
]
SANS_FACES = [
    "/System/Library/Fonts/Avenir Next.ttc",
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Supplemental/Futura.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "C:/Windows/Fonts/arial.ttf",
]

# Last resort: scan the usual font trees rather than failing outright.
FONT_DIRS = ["/System/Library/Fonts", "/Library/Fonts", "/usr/share/fonts",
             "/usr/local/share/fonts", "C:/Windows/Fonts"]

# Phrases that mean the user wants words rendered *in* the image.
TEXT_CUES = re.compile(
    r"""(?ix)
    \b(text|typography|lettering|wordmark|headline|caption|slogan|tagline|
       label\s+(?:reading|saying|with|that)|written|writing|signage|
       says|reading|spelled|font|logo\s*type)\b
    | "[^"]{2,}"            # a quoted string -- almost always words to render
    | '[^']{2,}'
    """,
)


def wants_text(prompt: str) -> bool:
    """True when the prompt asks for words inside the image."""
    return bool(TEXT_CUES.search(prompt))


def _first_existing(paths: list[str], serif: bool = False) -> str | None:
    found = next((p for p in paths if Path(p).exists()), None)
    if found:
        return found
    # Nothing from the curated list -- take any usable face from the font trees.
    want = ("serif", "roman", "georgia", "times") if serif else ("sans", "arial", "helvetica")
    for root in FONT_DIRS:
        base = Path(root)
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.tt[fc]")):
            name = path.name.lower()
            if any(w in name for w in want) and "italic" not in name and "bold" not in name:
                return str(path)
    return None


def available() -> bool:
    try:
        import PIL  # noqa: F401
    except ImportError:
        return False
    return bool(_first_existing(SERIF_FACES, serif=True) and _first_existing(SANS_FACES))


def build_ad(src: Path, out: Path, headline: str = "", kicker: str = "", sub: str = "",
             benefit: str = "", cta: str = "", size: tuple[int, int] | None = None) -> Path:
    """Upscale the render, add contrast scrims, and set type. Only draws what is given."""
    from PIL import Image, ImageDraw, ImageEnhance, ImageFont

    serif = _first_existing(SERIF_FACES, serif=True)
    sans = _first_existing(SANS_FACES)
    if not serif or not sans:
        raise RuntimeError("No suitable system fonts found for typesetting.")

    img = Image.open(src).convert("RGB")
    W, H = size or img.size
    img = img.resize((W, H), Image.LANCZOS)
    img = ImageEnhance.Contrast(img).enhance(1.06)

    top_needed = bool(headline or kicker or sub)
    bottom_needed = bool(benefit or cta)

    scrim = Image.new("L", (W, H), 0)
    sd = ImageDraw.Draw(scrim)
    if top_needed:
        depth = int(H * 0.32)
        for i in range(depth):
            sd.line([(0, i), (W, i)], fill=int(215 * (1 - (i / depth) ** 1.5)))
    if bottom_needed:
        depth = int(H * 0.22)
        for i in range(depth):
            y = H - 1 - i
            sd.line([(0, y), (W, y)], fill=int(185 * (1 - i / depth)))
    img = Image.composite(Image.new("RGB", (W, H), (6, 4, 3)), img, scrim)

    draw = ImageDraw.Draw(img)
    cx = W // 2

    def font(path: str, px: int):
        return ImageFont.truetype(path, px, index=0)

    def tracked(xy, text: str, f, fill, tracking: int) -> None:
        widths = [draw.textlength(c, font=f) for c in text]
        x = xy[0] - (sum(widths) + tracking * (len(text) - 1)) / 2
        for ch, w in zip(text, widths):
            draw.text((x, xy[1]), ch, font=f, fill=fill)
            x += w + tracking

    def fit(text: str, path: str, start: int, tracking: int, max_w: float):
        """Shrink until it fits -- a clipped headline is worse than a small one."""
        px = start
        while px > 8:
            f = font(path, px)
            width = sum(draw.textlength(c, font=f) for c in text) + tracking * (len(text) - 1)
            if width <= max_w:
                return f
            px -= 2
        return font(path, 8)

    def rule(y: int, half: int) -> None:
        draw.line([(cx - half, y), (cx + half, y)], fill=GOLD, width=2)

    y = int(H * 0.06)
    if kicker:
        tracked((cx, y), kicker, fit(kicker, sans, int(H * 0.022), 11, W * 0.7), GOLD, 11)
        y += int(H * 0.047)
    if headline:
        tracked((cx, y), headline, fit(headline, serif, int(H * 0.058), 4, W * 0.86), IVORY, 4)
        y += int(H * 0.08)
    if sub:
        rule(y, int(W * 0.11))
        y += int(H * 0.018)
        tracked((cx, y), sub, fit(sub, sans, int(H * 0.018), 7, W * 0.7), GOLD, 7)

    if benefit:
        tracked((cx, H - int(H * 0.19)), benefit,
                fit(benefit, sans, int(H * 0.019), 4, W * 0.84), IVORY, 4)
        rule(H - int(H * 0.145), int(W * 0.083))
    if cta:
        f = fit(cta, sans, int(H * 0.018), 6, W * 0.6)
        text_w = sum(draw.textlength(c, font=f) for c in cta) + 6 * (len(cta) - 1)
        bw, bh = text_w + 86, int(H * 0.049)
        top = H - int(H * 0.111)
        draw.rectangle([cx - bw // 2, top, cx + bw // 2, top + bh], outline=GOLD, width=2)
        tracked((cx, top + bh // 3), cta, f, GOLD, 6)

    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, "PNG", optimize=True)
    return out
