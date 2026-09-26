from pathlib import Path
import json
import re
import numpy as np
from sentence_transformers import SentenceTransformer

from config import DATA_DIR, VECTOR_DIR, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP

def read_file(path):
    ext = path.suffix.lower()

    if ext in [".txt", ".md"]:
        return path.read_text(encoding="utf-8", errors="ignore")

    if ext == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        pages = []
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            pages.append(f"[Page {i}]\n{text}")
        return "\n\n".join(pages)

    if ext == ".docx":
        from docx import Document
        doc = Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs)

    return ""

def normalize(text):
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(len(words), start + chunk_size)
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end == len(words):
            break
        start = max(0, end - overlap)
    return chunks

def main():
    data_dir = Path(DATA_DIR)
    vector_dir = Path(VECTOR_DIR)
    vector_dir.mkdir(parents=True, exist_ok=True)

    records = []
    supported = {".txt", ".md", ".pdf", ".docx"}

    for path in sorted(data_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in supported:
            raw = normalize(read_file(path))
            chunks = chunk_text(raw)
            for i, chunk in enumerate(chunks):
                records.append({
                    "source": path.name,
                    "section": f"chunk_{i+1}",
                    "text": chunk
                })

    if not records:
        raise RuntimeError("No supported documents found in the data/ folder.")

    embedder = SentenceTransformer(EMBEDDING_MODEL)
    texts = [r["text"] for r in records]
    embeddings = embedder.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True
    ).astype("float32")

    np.save(vector_dir / "embeddings.npy", embeddings)
    with open(vector_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"Indexed {len(records)} chunks from {len(set(r['source'] for r in records))} documents.")
    print(f"Saved to {vector_dir.resolve()}")

if __name__ == "__main__":
    main()
