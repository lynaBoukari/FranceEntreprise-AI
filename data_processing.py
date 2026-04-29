import os
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ─── CONFIGURATION ────────────────────────────────────────────────────────────

DOCUMENT_DIRECTORY = "Base_documentaire/"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
MIN_CHUNK_LENGTH = 200


# ─── CHARGEMENT ───────────────────────────────────────────────────────────────

def load_documents(directory: str = DOCUMENT_DIRECTORY) -> list:
    """Charge tous les PDFs depuis le dossier spécifié."""
    all_documents = []

    pdf_files = [f for f in os.listdir(directory) if f.endswith(".pdf")]
    print(f"📂 {len(pdf_files)} PDFs trouvés dans {directory}")

    for filename in pdf_files:
        file_path = os.path.join(directory, filename)
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        all_documents.extend(docs)
        print(f"   ✓ {filename} — {len(docs)} pages")

    print(f"\n📄 Total pages chargées : {len(all_documents)}")
    return all_documents


# ─── NETTOYAGE ────────────────────────────────────────────────────────────────

NOISE_PATTERNS = [
    "sommaire",
    "table des matières",
    "voir aussi",
    "en savoir plus",
    "accueil",
    "service-public.fr",
    "legifrance",
    "gouvernement.fr",
    "mise à jour",
    "publié le",
    "partager",
]

def clean_text(text: str) -> str:
    """Supprime les URLs et les espaces multiples."""
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"\s+", " ", text)

    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        line = line.strip()
        if len(line) < 20:
            continue
        if any(n in line.lower() for n in NOISE_PATTERNS):
            continue
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def remove_navigation(text: str) -> str:
    """Supprime les fils d'Ariane (Accueil > ... )."""
    return re.sub(r"Accueil\s*>.*", "", text)


def deduplicate_lines(text: str) -> str:
    """Supprime les lignes en double."""
    seen = set()
    result = []
    for line in text.split("\n"):
        if line not in seen:
            seen.add(line)
            result.append(line)
    return "\n".join(result)


def preprocess(text: str) -> str:
    """Pipeline complet de nettoyage d'un texte."""
    text = clean_text(text)
    text = remove_navigation(text)
    text = deduplicate_lines(text)
    return text


def clean_documents(documents: list) -> list:
    """Applique le nettoyage sur tous les documents."""
    cleaned = []
    for doc in documents:
        doc.page_content = preprocess(doc.page_content)
        cleaned.append(doc)
    print(f"🧹 Documents nettoyés : {len(cleaned)}")
    return cleaned


# ─── CHUNKING ─────────────────────────────────────────────────────────────────

def split_documents(documents: list) -> list:
    """Découpe les documents en chunks et filtre les trop courts."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "]
    )

    chunks = splitter.split_documents(documents)
    chunks_propres = [c for c in chunks if len(c.page_content.strip()) > MIN_CHUNK_LENGTH]

    print(f"✂️  Chunks avant nettoyage : {len(chunks)}")
    print(f"✅  Chunks après nettoyage : {len(chunks_propres)}")

    return chunks_propres


# ─── PIPELINE COMPLET ─────────────────────────────────────────────────────────

def build_chunks(directory: str = DOCUMENT_DIRECTORY) -> list:
    """Pipeline complet : charge → nettoie → découpe."""
    documents = load_documents(directory)
    documents = clean_documents(documents)
    chunks = split_documents(documents)
    return chunks


# ─── TEST DIRECT ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    chunks = build_chunks()
    print(f"\n🎯 Pipeline terminé — {len(chunks)} chunks prêts pour l'indexation.")