"""
Usage:
    python -m rag.ingest              # ingest everything
    python -m rag.ingest onet         # only O*NET
    python -m rag.ingest courses      # only courses
    python -m rag.ingest general      # only general docs (txt/pdf/md)
    python -m rag.ingest resume <uid> <path>  # one user resume
"""

import sys, json, hashlib, pathlib, textwrap

DOCS_ROOT = pathlib.Path(__file__).parent.parent / "docs"

# ── Chunker ──────────────────────────────────────────────────

def chunk_text(text: str, size: int = 400, overlap: int = 80) -> list[str]:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i:i + size]))
        i += size - overlap
    return [c for c in chunks if len(c.strip()) > 30]

def make_id(*parts) -> str:
    return hashlib.md5(":".join(str(p) for p in parts).encode()).hexdigest()

# ── O*NET ────────────────────────────────────────────────────

def ingest_onet():
    from rag.chroma_store import upsert
    path = DOCS_ROOT / "onet" / "careers.json"
    if not path.exists():
        print(f"[ingest] SKIP onet — {path} not found")
        return

    careers = json.loads(path.read_text())
    docs = []
    for c in careers:
        text = textwrap.dedent(f"""
            Career: {c.get('title', '')}
            SOC Code: {c.get('soc_code', '')}
            Description: {c.get('description', '')}
            Tasks: {', '.join(c.get('tasks', [])[:6])}
            Skills: {', '.join(c.get('skills', [])[:8])}
            Median Salary: ${c.get('median_salary', 'N/A')}
            Growth Outlook: {c.get('outlook', 'N/A')}
            Education: {c.get('education', 'N/A')}
        """).strip()

        docs.append({
            "id": c.get("soc_code", make_id(c.get("title", ""))),
            "text": text,
            "metadata": {
                "source": "onet",
                "title": c.get("title", ""),
                "soc_code": c.get("soc_code", ""),
                "median_salary": str(c.get("median_salary", "")),
                "outlook": c.get("outlook", ""),
            }
        })

    upsert("onet_careers", docs)
    print(f"[ingest] O*NET: {len(docs)} careers ingested")

# ── Courses ──────────────────────────────────────────────────

def ingest_courses():
    from rag.chroma_store import upsert
    path = DOCS_ROOT / "courses" / "courses.json"
    if not path.exists():
        print(f"[ingest] SKIP courses — {path} not found")
        return

    courses = json.loads(path.read_text())
    docs = []
    for c in courses:
        text = textwrap.dedent(f"""
            Course: {c.get('title', '')}
            Platform: {c.get('platform', '')}
            Provider: {c.get('provider', '')}
            Description: {c.get('description', '')}
            Skills covered: {', '.join(c.get('skills', []))}
            Level: {c.get('level', '')}
            Duration: {c.get('duration', 'N/A')}
        """).strip()

        docs.append({
            "id": c.get("id", make_id(c.get("title", ""), c.get("platform", ""))),
            "text": text,
            "metadata": {
                "source": "course_catalog",
                "title": c.get("title", ""),
                "platform": c.get("platform", ""),
                "url": c.get("url", ""),
                "level": c.get("level", ""),
                "skills": ", ".join(c.get("skills", [])),
            }
        })

    upsert("course_catalog", docs)
    print(f"[ingest] Courses: {len(docs)} courses ingested")

# ── General docs (txt / md / pdf) ────────────────────────────

def ingest_general():
    from rag.chroma_store import upsert
    general_dir = DOCS_ROOT / "general"
    if not general_dir.exists():
        print("[ingest] SKIP general — docs/general/ not found")
        return

    docs = []
    for fpath in general_dir.rglob("*"):
        if fpath.suffix not in (".txt", ".md", ".pdf"):
            continue
        text = _read_file(fpath)
        if not text:
            continue
        for i, chunk in enumerate(chunk_text(text)):
            docs.append({
                "id": make_id(fpath.name, i),
                "text": chunk,
                "metadata": {
                    "source": "general_docs",
                    "filename": fpath.name,
                    "chunk": i,
                }
            })

    upsert("general_docs", docs)
    print(f"[ingest] General docs: {len(docs)} chunks ingested")

# ── User resume / transcript ──────────────────────────────────

def ingest_resume(user_id: str, file_path: str):
    from rag.chroma_store import upsert
    path = pathlib.Path(file_path)
    if not path.exists():
        print(f"[ingest] ERROR — file not found: {file_path}")
        return

    text = _read_file(path)
    if not text:
        print("[ingest] ERROR — could not read file")
        return

    docs = []
    for i, chunk in enumerate(chunk_text(text, size=300, overlap=60)):
        docs.append({
            "id": make_id(user_id, path.name, i),
            "text": chunk,
            "metadata": {
                "source": "user_document",
                "user_id": user_id,
                "filename": path.name,
                "chunk": i,
            }
        })

    upsert("user_documents", docs)
    print(f"[ingest] Resume for {user_id}: {len(docs)} chunks ingested")

# ── File reader ───────────────────────────────────────────────

def _read_file(path: pathlib.Path) -> str:
    if path.suffix == ".pdf":
        try:
            import pypdf
            reader = pypdf.PdfReader(str(path))
            return " ".join(p.extract_text() or "" for p in reader.pages)
        except ImportError:
            print("[ingest] WARN — pypdf not installed, skipping PDF. pip install pypdf")
            return ""
    else:
        return path.read_text(errors="replace")

# ── CLI entry ─────────────────────────────────────────────────

if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] == "all":
        ingest_onet()
        ingest_courses()
        ingest_general()
    elif args[0] == "onet":
        ingest_onet()
    elif args[0] == "courses":
        ingest_courses()
    elif args[0] == "general":
        ingest_general()
    elif args[0] == "resume" and len(args) == 3:
        ingest_resume(user_id=args[1], file_path=args[2])
    else:
        print(__doc__)