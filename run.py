"""Top-level entry point for frozen (Nuitka) builds.

Nuitka compiles a program most reliably when the entry script sits at the repo
root and imports the ``app`` package *by name* — exactly what ``python -m app.main``
does. Pointing Nuitka straight at ``app/main.py`` instead makes it treat the file
as a loose top-level script and mis-resolve the absolute ``from app...`` imports.

Running this directly is equivalent to ``python -m app.main``.
"""
import sys

from app.main import main

if __name__ == "__main__":
    sys.exit(main())
