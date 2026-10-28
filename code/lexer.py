"""Regex-based lexer for the mini language used throughout the project."""
import re

KEYWORDS = {"if", "else", "while", "print"}
_SPEC = [
    ("ws",   r"\s+"),
    ("num",  r"\d+"),
    ("id",   r"[A-Za-z_][A-Za-z_0-9]*"),
    ("op2",  r"==|!="),
    ("op1",  r"[+\-*/=<>(){};,]"),
]
_RE = re.compile("|".join(f"(?P<{n}>{p})" for n, p in _SPEC))


class LexError(Exception):
    pass


def tokenize(src):
    """Return list of (terminal, lexeme). Terminal names match grammar symbols."""
    out, pos = [], 0
    while pos < len(src):
        m = _RE.match(src, pos)
        if not m:
            raise LexError(f"Illegal character {src[pos]!r} at offset {pos}")
        kind, text = m.lastgroup, m.group()
        pos = m.end()
        if kind == "ws":
            continue
        if kind == "id" and text in KEYWORDS:
            out.append((text, text))
        elif kind in ("id", "num"):
            out.append((kind, text))
        else:
            out.append((text, text))
    return out
