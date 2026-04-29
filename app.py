import streamlit as st
from dotenv import load_dotenv
from rag_pipeline import init_pipeline, repondre

# ─── CONFIGURATION ────────────────────────────────────────────────────────────

load_dotenv()

st.set_page_config(
    page_title="FranceEntreprise AI",
    page_icon="🇫🇷",
    layout="centered"
)

# ─── TEXTES MULTILINGUES ──────────────────────────────────────────────────────

TEXTS = {
    "Français": {
        "title": "🇫🇷 FranceEntreprise AI",
        "subtitle": "Votre assistant expert en entrepreneuriat en France",
        "placeholder": "Posez votre question sur l'entrepreneuriat en France...",
        "thinking": "L'assistant réfléchit...",
        "sources": "📄 Sources utilisées",
        "no_sources": "Aucune source trouvée",
        "welcome": "Bonjour ! Je suis votre assistant expert en entrepreneuriat en France. Posez-moi vos questions sur la création d'entreprise, le statut auto-entrepreneur, les charges, les aides disponibles, etc.",
        "language_label": "Langue / Language",
    },
    "English": {
        "title": "🇫🇷 FranceEntreprise AI",
        "subtitle": "Your expert assistant on entrepreneurship in France",
        "placeholder": "Ask your question about entrepreneurship in France...",
        "thinking": "The assistant is thinking...",
        "sources": "📄 Sources used",
        "no_sources": "No sources found",
        "welcome": "Hello! I am your expert assistant on entrepreneurship in France. Ask me anything about business creation, self-employment status, taxes, available grants, etc.",
        "language_label": "Langue / Language",
    }
}

# ─── CHARGEMENT DU PIPELINE (une seule fois) ──────────────────────────────────

@st.cache_resource
def load_pipeline():
    """Charge le pipeline RAG une seule fois et le garde en mémoire."""
    return init_pipeline()

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.sidebar.markdown("# 🇫🇷")
    st.markdown("---")

    langue = st.radio(
        "Langue / Language",
        options=["Français", "English"],
        index=0
    )

    texts = TEXTS[langue]

    st.markdown("---")
    st.markdown("### 📚 Documents indexés")
    st.markdown("""
    - Guide pratique BPI France
    - Guide Auto-Entrepreneur 2023
    - Micro-entreprise — economie.gouv.fr
    - Création d'entreprise — Service Public
    - Immatriculation micro-entreprise
    - Et plus...
    """)

    st.markdown("---")

    if st.button("🗑️ Effacer la conversation" if langue == "Français" else "🗑️ Clear conversation"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.caption("Propulsé par LangChain · ChromaDB · Groq · Llama 3.3")

# ─── TITRE PRINCIPAL ──────────────────────────────────────────────────────────

st.title(texts["title"])
st.caption(texts["subtitle"])
st.markdown("---")

# ─── INITIALISATION DE L'HISTORIQUE ──────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": texts["welcome"],
            "sources": []
        }
    ]

# ─── CHARGEMENT DU PIPELINE ───────────────────────────────────────────────────

with st.spinner("⚙️ Chargement de l'assistant..." if langue == "Français" else "⚙️ Loading assistant..."):
    rag_chain, retriever = load_pipeline()

# ─── AFFICHAGE DE L'HISTORIQUE ────────────────────────────────────────────────

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

        # Afficher les sources si disponibles
        if message.get("sources"):
            with st.expander(texts["sources"]):
                for source in message["sources"]:
                    st.markdown(f"- 📄 `{source}`")

# ─── SAISIE DE LA QUESTION ────────────────────────────────────────────────────

question = st.chat_input(texts["placeholder"])

if question:
    # Afficher la question de l'utilisateur
    with st.chat_message("user"):
        st.write(question)

    # Ajouter à l'historique
    st.session_state.messages.append({
        "role": "user",
        "content": question,
        "sources": []
    })

    # Appeler le RAG et afficher la réponse
    with st.chat_message("assistant"):
        with st.spinner(texts["thinking"]):
            result = repondre(question, rag_chain, retriever)

        st.write(result["reponse"])

        # Afficher les sources
        if result["sources"]:
            with st.expander(texts["sources"]):
                for source in result["sources"]:
                    st.markdown(f"- 📄 `{source}`")
        else:
            st.caption(texts["no_sources"])

    # Ajouter la réponse à l'historique
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["reponse"],
        "sources": result["sources"]
    })