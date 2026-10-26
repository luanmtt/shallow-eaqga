import math
import os
import sys


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# Gate de cor: ligada só em terminal interativo (logs em arquivo saem limpos).
# Respeita NO_COLOR (padrão: presente e não-vazio desliga) e permite override
# via set_color_enabled() (ex: futuras flags --color/--no-color).

_color_override = None  # None = automático


def set_color_enabled(enabled) -> None:
    """Força cor ligada/desligada. None (default) = automático."""
    global _color_override
    _color_override = None if enabled is None else bool(enabled)


def colors_enabled() -> bool:
    if _color_override is not None:
        return _color_override
    if os.environ.get("NO_COLOR"):
        return False
    isatty = getattr(sys.stdout, "isatty", None)
    return bool(isatty and isatty())


# ──────────────────────────────────────────────────────────────────────────────────────────────────


def sep(n: int) -> str:
    return n * "─" + "\n"
 

def sepComment(n: int, comment: str) -> str:
    
    result = (math.floor(n/2) * "─") + " " + comment + " " + (math.ceil(n/2) * "─") + "\n"
    return result


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# cores ANSI:


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def colorPrint(text: str, hex_color: str) -> str:
    if not colors_enabled():
        return text
    r, g, b = hex_to_rgb(hex_color)
    return f"\033[38;2;{r};{g};{b}m{text}\033[0m"


def stylePrint(text: str, hex_color: str = None, bold: bool = False, italic: bool = False, underline: bool = False) -> str:
    if not colors_enabled():
        return text
    codes = []
    if bold:      codes.append("1")
    if italic:    codes.append("3")
    if underline: codes.append("4")
    if hex_color:
        r, g, b = hex_to_rgb(hex_color)
        codes.append(f"38;2;{r};{g};{b}")
    if not codes:
        return text
    return f"\033[{';'.join(codes)}m{text}\033[0m"


# ──────────────────────────────────────────────────────────────────────────────────────────────────
