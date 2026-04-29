import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from data_processing import build_chunks

# ─── CONFIGURATION ────────────────────────────────────────────────────────────

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL = "llama-3.3-70b-versatile"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # chargé depuis .env
K_RETRIEVER = 6

PROMPT_TEMPLATE = """Tu es un assistant expert en entrepreneuriat en France.
Tu as un ton très professionnel, tu expliques bien et vulgarises l'information.
Utilise uniquement les extraits suivants pour répondre.
Si la réponse n'est pas dans les extraits, dis "Je ne trouve pas cette information dans les documents."

Extraits : {context}
Question : {question}
Réponse :"""


# ─── CONSTRUCTION DU PIPELINE ─────────────────────────────────────────────────

def build_vectorstore(chunks: list) -> Chroma:
    """Crée les embeddings et indexe les chunks dans ChromaDB."""
    print("⚙️  Chargement du modèle d'embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print("📦 Indexation dans ChromaDB...")
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
    print("✅ Indexation terminée !")
    return vectorstore


def build_retriever(vectorstore: Chroma):
    """Crée le retriever à partir du vectorstore."""
    return vectorstore.as_retriever(search_kwargs={"k": K_RETRIEVER})


def build_llm() -> ChatGroq:
    """Initialise le LLM Groq."""
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=LLM_MODEL,
        temperature=0
    )


def build_rag_chain(retriever, llm):
    """Assemble la chain RAG complète."""
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


# ─── FONCTION PRINCIPALE ──────────────────────────────────────────────────────

def init_pipeline():
    """
    Pipeline complet : charge les docs → indexe → construit la chain RAG.
    Retourne (rag_chain, retriever) .
    """
    chunks = build_chunks()
    vectorstore = build_vectorstore(chunks)
    retriever = build_retriever(vectorstore)
    llm = build_llm()
    rag_chain = build_rag_chain(retriever, llm)
    print("\n🤖 Agent RAG prêt !")
    return rag_chain, retriever


# ─── RÉPONSE AVEC SOURCES ─────────────────────────────────────────────────────

def repondre(question: str, rag_chain, retriever) -> dict:
    """
    Pose une question à l'agent et retourne la réponse + les sources.
    Retourne un dict : {"reponse": str, "sources": list}
    """
    docs = retriever.invoke(question)
    sources = list(set([
        os.path.basename(doc.metadata.get("source", "inconnu"))
        for doc in docs
    ]))
    reponse = rag_chain.invoke(question)
    return {"reponse": reponse, "sources": sources}


# ─── TEST DIRECT ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    rag_chain, retriever = init_pipeline()
    result = repondre(
        "Quelles sont les charges d'un auto-entrepreneur en France ?",
        rag_chain,
        retriever
    )
    print(f"\nRéponse : {result['reponse']}")
    print(f"\nSources : {result['sources']}")