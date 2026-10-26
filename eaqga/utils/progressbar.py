import sys
import time
import threading
from .embellish import colors_enabled

def _color_to_ansi(color: str | None) -> str | None:
    """Convert a hex color to an ANSI foreground sequence, or pass ANSI through."""

    if color is None:
        return None
    if color.startswith("\033["):
        return color
    h = color.lstrip("#")
    r, g, b = (int(h[i:i+2], 16) for i in (0, 2, 4))
    return f"\033[38;2;{r};{g};{b}m"


class ProgressBar:
    """
    A terminal progress bar supporting keras and npm styles.

    Usage:
        bar = ProgressBar(total=312, style="keras")
        bar.update(100)
        bar.set_suffix("loss: 2.54 - accuracy: 0.26")
        bar.update(200)
        bar.close()
    """

    BRAILLE = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    BAR_WIDTH = 20
    RESET = "\033[0m"

    DEFAULT_FG = {
        "keras": "\033[38;5;43m",
        "npm": "\033[38;5;160m",
    }
    DEFAULT_BG = {
        "keras": "\033[2m",
        "npm": "\033[38;5;238m",
    }

    def __init__(self, total: int, style: str = "keras",
                 fg_color: str | None = None, bg_color: str | None = None):
        """
        Args:
            total: Maximum value.
            style: "keras" or "npm".
            fg_color: ANSI color for the filled part.
            bg_color: ANSI color for the empty part.
        """
        assert style in ("keras", "npm"), "style must be 'keras' or 'npm'"
        self.total = total
        self.style = style
        # Sem terminal (log em arquivo): barras sem ANSI (símbolos preservados).
        use_color = colors_enabled()
        if fg_color is not None:
            self.fg_color = _color_to_ansi(fg_color) if use_color else ""
        else:
            self.fg_color = self.DEFAULT_FG[style] if use_color else ""
        if bg_color is not None:
            self.bg_color = _color_to_ansi(bg_color) if use_color else ""
        else:
            self.bg_color = self.DEFAULT_BG[style] if use_color else ""
        if not use_color:
            self.RESET = ""  # sombreia a constante de classe
        self.current = 0
        self.suffix = ""
        self._spinner_frame = 0
        self._lock = threading.Lock()
        self._spinner_thread = None
        self._running = False

        if style == "npm":
            self._running = True
            self._spinner_thread = threading.Thread(target=self._spin, daemon=True)
            self._spinner_thread.start()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def update(self, value: int):
        """Set the current progress value and redraw."""
        with self._lock:
            self.current = min(value, self.total)
        if self.style == "keras":
            self._draw()

    def set_suffix(self, text: str):
        """Append arbitrary text to the right of the bar."""
        with self._lock:
            self.suffix = text

    def close(self):
        """Finalise the bar and move to a new line."""
        self._running = False
        if self._spinner_thread:
            self._spinner_thread.join()
        with self._lock:
            self.current = self.total
        self._draw(end="\n")

    def __getstate__(self):
        """Exclude non-picklable lock/thread state for multiprocessing."""
        state = self.__dict__.copy()
        state['_lock'] = None
        state['_spinner_thread'] = None
        state['_running'] = False
        return state

    def __setstate__(self, state):
        """Restore a fresh lock when unpickled."""
        self.__dict__.update(state)
        self._lock = threading.Lock()
        self._spinner_thread = None
        if self.style == "npm":
            self._running = True
            self._spinner_thread = threading.Thread(target=self._spin, daemon=True)
            self._spinner_thread.start()

    def bar(self, ratio: float | None = None) -> str:
        """Return only the coloured bar string for a given ratio (or current state)."""
        if ratio is None:
            with self._lock:
                ratio = self.current / self.total if self.total else 0.0
        filled = max(0, min(self.BAR_WIDTH, int(round(ratio * self.BAR_WIDTH))))
        empty = self.BAR_WIDTH - filled
        fill_char = "█" if self.style == "keras" else "█"
        parts = []
        if filled:
            parts.append(f"{self.fg_color}{fill_char * filled}{self.RESET}")
        if empty:
            parts.append(f"{self.bg_color}{'░' * empty}{self.RESET}")
        return "".join(parts)

    def render(self, current: int | None = None, total: int | None = None) -> str:
        """Return the full progress line without printing it."""
        if current is None:
            with self._lock:
                current = self.current
        if total is None:
            total = self.total

        pct = current / total if total else 0
        width = len(str(total))
        count = f"{str(current).rjust(width)}/{total}"
        bar = self.bar(pct)
        suffix_str = f"  {self.suffix}" if self.suffix else ""

        if self.style == "keras":
            return f"{count} {bar}{suffix_str}"

        spinner = self.BRAILLE[self._spinner_frame]
        pct_str = f"{round(pct * 100):>3}%"
        return f"{self.fg_color}{spinner}{self.RESET} {bar} {pct_str}{suffix_str}"

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _spin(self):
        while self._running:
            self._draw()
            self._spinner_frame = (self._spinner_frame + 1) % len(self.BRAILLE)
            time.sleep(0.08)

    def _draw(self, end="\r"):
        line = self.render()
        sys.stdout.write(f"\r{line}    ")
        sys.stdout.flush()
        if end == "\n":
            sys.stdout.write("\n")
            sys.stdout.flush()
