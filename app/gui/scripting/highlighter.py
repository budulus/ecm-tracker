"""Python syntax highlighting for the Script Editor and the console (light theme colors)."""
import builtins
import keyword
import re

from PySide6.QtGui import QColor, QFont, QSyntaxHighlighter, QTextCharFormat


def _fmt(color: str, bold: bool = False, italic: bool = False) -> QTextCharFormat:
    fmt = QTextCharFormat()
    fmt.setForeground(QColor(color))
    if bold:
        fmt.setFontWeight(QFont.Weight.Bold)
    if italic:
        fmt.setFontItalic(True)
    return fmt


STYLES = {
    "keyword": _fmt("#7c3aed", bold=True),
    "builtin": _fmt("#0e7490"),
    "self": _fmt("#9a3412", italic=True),
    "api": _fmt("#2563eb", bold=True),  # session / app / np: the scripting entry points
    "number": _fmt("#b45309"),
    "string": _fmt("#15803d"),
    "comment": _fmt("#8a9099", italic=True),
    "decorator": _fmt("#a16207"),
    "defname": _fmt("#1d4ed8", bold=True),
}

_BUILTINS = sorted(n for n in dir(builtins) if not n.startswith("_"))
_RULES = [
    (re.compile(r"\b(?:%s)\b" % "|".join(keyword.kwlist + list(keyword.softkwlist))), "keyword"),
    (re.compile(r"\b(?:%s)\b" % "|".join(map(re.escape, _BUILTINS))), "builtin"),
    (re.compile(r"\b(?:session|app|np)\b"), "api"),
    (re.compile(r"\b(?:self|cls)\b"), "self"),
    (re.compile(r"\b(?:0[xob][0-9a-fA-F_]+|\d[\d_]*\.?\d*(?:[eE][+-]?\d+)?j?)\b"), "number"),
    (re.compile(r"^\s*@[\w.]+"), "decorator"),
    (re.compile(r"\b(?:def|class)\s+(\w+)"), "defname"),
]
# Single-line strings (with optional prefixes) and comments; scanned left to right so a '#'
# inside a string is not a comment and quotes inside a comment are not strings.
_STRING_OR_COMMENT = re.compile(
    r"""(?P<string>\b[rRbBuUfF]{0,2}(?:'(?:[^'\\\n]|\\.)*'?|"(?:[^"\\\n]|\\.)*"?))|(?P<comment>\#.*)"""
)
_TRIPLE = (re.compile(r"'''"), re.compile(r'"""'))


class PythonHighlighter(QSyntaxHighlighter):
    """Keywords, builtins, numbers, strings (incl. multi-line triple quotes) and comments."""

    def highlightBlock(self, text: str) -> None:
        for pattern, style in _RULES:
            for match in pattern.finditer(text):
                group = 1 if match.re.groups else 0
                start, end = match.span(group)
                self.setFormat(start, end - start, STYLES[style])
        for match in _STRING_OR_COMMENT.finditer(text):
            kind = "string" if match.group("string") else "comment"
            self.setFormat(match.start(), match.end() - match.start(), STYLES[kind])
        self._triple_quotes(text)  # last: a triple-quoted span overrides the line-local rules

    def _triple_quotes(self, text: str) -> None:
        """Multi-line strings via block state: 0 = normal, 1 = inside ''' , 2 = inside \"\"\"."""
        self.setCurrentBlockState(0)
        start = 0
        state = self.previousBlockState()
        if state in (1, 2):
            delimiter = _TRIPLE[state - 1]
        else:
            delimiter = None
            for index, candidate in enumerate(_TRIPLE, start=1):
                match = candidate.search(text)
                if match and (delimiter is None or match.start() < start):
                    delimiter, state, start = candidate, index, match.start()
            if delimiter is None:
                return
            start_offset = start + 3
            match = delimiter.search(text, start_offset)
            if match:
                self.setFormat(start, match.end() - start, STYLES["string"])
                return
            self.setCurrentBlockState(state)
            self.setFormat(start, len(text) - start, STYLES["string"])
            return
        match = delimiter.search(text)
        if match:
            self.setFormat(0, match.end(), STYLES["string"])
        else:
            self.setCurrentBlockState(state)
            self.setFormat(0, len(text), STYLES["string"])
