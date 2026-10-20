"""``python -m ecm_tracker script.py [args...]`` runs a script headless."""
import sys

from app.scripting.cli import main

sys.exit(main(sys.argv[1:]))
