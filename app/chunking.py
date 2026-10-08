"""Converte páginas Markdown do site em trechos (chunks) prontos para indexar."""
import re
from dataclasses import dataclass, asdict

from .config import SITE_URL

MAX_CHARS = 1400
MIN_CHARS = 250

_HEADING = re.compile(r"^(#{1,3})\s+(.*\S)\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")
_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_ATTR = re.compile(r"\{[:\s]*[.#][^}]*\}")  # {: .md-button } do mkdocs
_FRONT_MATTER = re.compile(r"\A---\n.*?\n---\n", re.S)


@dataclass
class Chunk:
    id: str
    source: str  # caminho relativo ao docs_dir, ex.: aulas/genAI/lab4/lab4.md
    url: str
    title: str  # título da página
    section: str  # trilha de títulos, ex.: "Embeddings > Trechos e similaridade"
    text: str

    def embed_text(self) -> str:
        """Texto enviado ao modelo de embeddings: o contexto do título ajuda a busca."""
        return f"{self.title} > {self.section}\n{self.text}"

    def to_dict(self) -> dict:
        return asdict(self)


def page_url(source: str) -> str:
    """aulas/genAI/lab4/lab4.md -> .../aulas/genAI/lab4/lab4/ (mkdocs use_directory_urls)."""
    path = source[:-3] if source.endswith(".md") else source
    if path == "index":
        return SITE_URL
    if path.endswith("/index"):
        path = path[: -len("index")]
    else:
        path += "/"
    return SITE_URL + path


def clean(md: str) -> str:
    md = md.replace("\r\n", "\n")
    md = _FRONT_MATTER.sub("", md, count=1)
    md = _IMAGE.sub("", md)
    md = _ATTR.sub("", md)
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()


def _sections(md: str, fallback_title: str):
    """Quebra o Markdown em seções (título, trilha, texto) respeitando blocos de código."""
    title = fallback_title
    trail: list[str] = []
    buf: list[str] = []
    in_fence = False
    out = []

    def flush():
        text = "\n".join(buf).strip()
        if text:
            out.append((" > ".join(trail) or title, text))
        buf.clear()

    for line in md.split("\n"):
        if _FENCE.match(line):
            in_fence = not in_fence
        m = None if in_fence else _HEADING.match(line)
        if m:
            flush()
            level, name = len(m.group(1)), m.group(2).strip()
            if level == 1 and not out and not buf and title == fallback_title:
                title = name
            trail[:] = trail[: level - 1] + [name]
            continue
        buf.append(line)
    flush()
    return title, out


def _split_long(text: str) -> list[str]:
    """Divide um texto longo por parágrafos (blocos de código ficam inteiros se possível)."""
    blocks, cur, in_fence = [], [], False
    for line in text.split("\n"):
        if _FENCE.match(line):
            in_fence = not in_fence
        cur.append(line)
        if not in_fence and line.strip() == "":
            blocks.append("\n".join(cur).strip())
            cur = []
    if cur:
        blocks.append("\n".join(cur).strip())

    parts, acc = [], ""
    for b in filter(None, blocks):
        # bloco gigante (ex.: código enorme): corta por linhas
        while len(b) > MAX_CHARS * 1.5:
            cut = b.rfind("\n", 0, MAX_CHARS)
            cut = cut if cut > 0 else MAX_CHARS
            if acc:
                parts.append(acc)
                acc = ""
            parts.append(b[:cut].strip())
            b = b[cut:].strip()
        if acc and len(acc) + len(b) + 2 > MAX_CHARS:
            parts.append(acc)
            acc = b
        else:
            acc = f"{acc}\n\n{b}" if acc else b
    if acc:
        parts.append(acc)
    return parts


def chunk_page(source: str, markdown: str) -> list[Chunk]:
    fallback = source.rsplit("/", 1)[-1].removesuffix(".md").replace("_", " ")
    title, sections = _sections(clean(markdown), fallback)
    url = page_url(source)

    # junta seções muito curtas na seguinte, para não gerar trechos sem contexto
    merged: list[tuple[str, str]] = []
    pending_head, pending_text = None, ""
    for head, text in sections:
        if pending_text:
            text = f"{pending_text}\n\n{text}"  # mantém o título da seção que traz o conteúdo principal
        if len(text) < MIN_CHARS:
            pending_head, pending_text = head, text
            continue
        merged.append((head, text))
        pending_head, pending_text = None, ""
    if pending_text:
        if merged:
            h, t = merged[-1]
            merged[-1] = (h, f"{t}\n\n{pending_text}")
        else:
            merged.append((pending_head or title, pending_text))

    chunks = []
    for head, text in merged:
        for part in _split_long(text) if len(text) > MAX_CHARS else [text]:
            chunks.append(
                Chunk(
                    id=f"{source}#{len(chunks)}",
                    source=source,
                    url=url,
                    title=title,
                    section=head,
                    text=part,
                )
            )
    return chunks
