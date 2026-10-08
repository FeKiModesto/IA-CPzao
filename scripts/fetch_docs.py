"""Baixa os .md e os notebooks .ipynb do site da matéria (repositório público do professor) para data/raw/.

Usa o zip do GitHub em vez de `git clone`, porque o repositório tem nomes de
arquivo que o Windows não aceita no checkout.
"""
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import DOCS_ZIP_URL, RAW_DIR  # noqa: E402

PREFIX = "material/"
MAX_CODE_CELL = 3000  # células de código gigantes (dados embutidos etc.) são cortadas


def notebook_to_markdown(raw: bytes) -> str:
    """Texto das células markdown + código (sem saídas). O site publica cada notebook como página."""
    nb = json.loads(raw.decode("utf-8"))
    parts = []
    for cell in nb.get("cells", []):
        src = "".join(cell.get("source", [])).strip()
        if not src:
            continue
        if cell.get("cell_type") == "code":
            parts.append(f"```python\n{src[:MAX_CODE_CELL]}\n```")
        elif cell.get("cell_type") == "markdown":
            parts.append(src)
    return "\n\n".join(parts)


def main() -> int:
    print(f"Baixando {DOCS_ZIP_URL} ...")
    with urllib.request.urlopen(DOCS_ZIP_URL, timeout=120) as resp:
        data = resp.read()

    pages = notebooks = 0
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for info in zf.infolist():
            # entradas vêm como "<repo>-<branch>/material/..."
            _, _, rel = info.filename.partition("/")
            if not rel.startswith(PREFIX) or not rel.endswith((".md", ".ipynb")):
                continue
            rel = rel[len(PREFIX):]
            dest = RAW_DIR / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            if rel.endswith(".ipynb"):
                # vira "<nome>.md": a URL da página do notebook no site é a mesma de um .md
                dest = dest.with_suffix(".md")
                if dest.exists():
                    continue  # já existe uma página .md com esse nome; ela prevalece
                dest.write_text(notebook_to_markdown(zf.read(info)), encoding="utf-8")
                notebooks += 1
            else:
                dest.write_bytes(zf.read(info))
                pages += 1
    print(f"{pages} páginas e {notebooks} notebooks salvos em {RAW_DIR}")
    return 0 if pages else 1


if __name__ == "__main__":
    raise SystemExit(main())
