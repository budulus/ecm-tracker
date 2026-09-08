"""Generate the offline SDK reference directly from public source signatures/docstrings."""
from __future__ import annotations
import ast
import html
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def reference_text():
    sections = []
    for module in ("api", "analysis", "testing"):
        tree = ast.parse((ROOT / "app/plugins" / (module + ".py")).read_text(encoding="utf-8"))
        sections.append("app.plugins." + module + "\n\n" + (ast.get_docstring(tree) or ""))
        for node in tree.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and not node.name.startswith("_"):
                signature = node.name if isinstance(node, ast.ClassDef) else node.name + "(" + ast.unparse(node.args) + ")"
                sections.append(signature + "\n" + (ast.get_docstring(node) or ""))
                if isinstance(node, ast.ClassDef):
                    for member in node.body:
                        if isinstance(member, ast.AnnAssign):
                            sections.append("  " + ast.unparse(member))
                        if isinstance(member, ast.FunctionDef) and not member.name.startswith("_"):
                            returns = " -> " + ast.unparse(member.returns) if member.returns else ""
                            sections.append("  " + member.name + "(" + ast.unparse(member.args) + ")" + returns
                                            + "\n" + (ast.get_docstring(member) or ""))
    return "\n\n".join(sections)


def main():
    guide = (ROOT / "plugins/PLUGIN_CONTRACT.md").read_text(encoding="utf-8")
    reference = reference_text()
    page = ('<!doctype html><html lang="en"><meta charset="utf-8"><title>ECM Tracker SDK 1</title>'
            '<style>body{max-width:1050px;margin:40px auto;padding:0 24px;font:16px system-ui}'
            'pre{white-space:pre-wrap;line-height:1.6}h1{font-size:2em}a{color:#2563eb}</style>'
            '<h1>ECM Tracker SDK 1</h1><p>Generated from the contract and public API. '
            '<a href="PLUGIN_CONTRACT.md">Author guide</a> · <a href="#reference">API reference</a></p>'
            '<pre>' + html.escape(guide) + '</pre><h1 id="reference">Public API reference</h1><pre>'
            + html.escape(reference) + '</pre></html>\n')
    (ROOT / "plugins/plugin-api.html").write_text(page, encoding="utf-8")
    kit = ROOT / "plugins/_sdk"
    kit.mkdir(exist_ok=True)
    (kit / "API_REFERENCE.txt").write_text(reference, encoding="utf-8")
    shutil.copyfile(ROOT / "uv.lock", kit / "uv.lock")
    import importlib.metadata
    packages = ("numpy", "opencv-python", "PySide6", "scipy", "matplotlib")
    (kit / "host_dependencies.json").write_text(
        json.dumps({name: importlib.metadata.version(name) for name in packages}, indent=2) + "\n",
        encoding="utf-8")
    print("Generated offline SDK reference and dependency manifest.")


if __name__ == "__main__":
    main()
