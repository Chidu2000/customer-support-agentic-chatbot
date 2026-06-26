import sys
from pathlib import Path

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.graph import ask
from src.config import settings
from src.data.sample_policies import write_sample_policies
from src.ingestion.pdf_processor import ingest_file
from src.vectorstore.chroma_store import add_documents, clear_collection, list_indexed_sources

st.set_page_config(page_title="Customer Support Assistant", page_icon="💬", layout="wide")

ROUTE_BADGE = {
    "sql":    ("🗄️ SQL",    "#1a73e8", "#e8f0fe"),
    "rag":    ("📄 RAG",    "#188038", "#e6f4ea"),
    "hybrid": ("⚡ Hybrid", "#e37400", "#fef3e2"),
}


def route_badge_html(route: str) -> str:
    label, color, bg = ROUTE_BADGE.get(route, ("❓ Unknown", "#666", "#eee"))
    return (
        f'<span style="background:{bg};color:{color};border:1px solid {color};'
        f'border-radius:12px;padding:2px 10px;font-size:0.75rem;font-weight:600;'
        f'letter-spacing:0.03em">{label}</span>'
    )


def render_citations(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander("📚 Sources", expanded=False):
        for i, src in enumerate(sources, 1):
            fname = src.get("source_file", "unknown")
            page = src.get("page")
            page_str = f", p. {page + 1}" if page is not None else ""
            st.markdown(f"**[{i}]** `{fname}`{page_str}")


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Setup")
    st.caption(f"Model: `{settings.openai_model}`")

    # ── Upload ────────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📤 Upload policy document")
    st.caption("Upload a PDF or text file to add it to the knowledge base.")
    uploaded = st.file_uploader("Choose a file", type=["pdf", "txt", "md"], label_visibility="collapsed")
    if uploaded is not None:
        policies_dir = Path(settings.policies_dir)
        policies_dir.mkdir(parents=True, exist_ok=True)
        save_path = policies_dir / uploaded.name
        save_path.write_bytes(uploaded.getvalue())
        try:
            with st.spinner(f"Indexing {uploaded.name}…"):
                chunks = ingest_file(save_path)
                count = add_documents(chunks)
            st.success(f"✅ Indexed {count} chunks from **{uploaded.name}**")
            st.rerun()
        except Exception as exc:
            st.error(f"Ingestion failed: {exc}")

    # ── Knowledge base status ─────────────────────────────────────────────────
    st.divider()
    st.subheader("📚 Knowledge base")
    sources = list_indexed_sources()
    if sources:
        for source in sources:
            st.markdown(f"- `{source}`")
    else:
        st.info("No documents indexed yet.\nUpload a file above or load the sample policies below.")

    # ── Sample policies shortcut ──────────────────────────────────────────────
    st.divider()
    st.subheader("🧪 Demo data")
    st.caption("Load built-in sample policies (refund, shipping, privacy, warranty).")
    if st.button("Load sample policies", use_container_width=True):
        with st.spinner("Writing and indexing sample policies…"):
            try:
                paths = write_sample_policies()
                total = 0
                for path in paths:
                    chunks = ingest_file(path)
                    total += add_documents(chunks)
                st.success(f"✅ Loaded {len(paths)} sample policies ({total} chunks)")
                st.rerun()
            except Exception as exc:
                st.error(f"Failed: {exc}")

    if sources and st.button("🗑️ Clear index", use_container_width=True, type="secondary"):
        removed = clear_collection()
        st.warning(f"Cleared {removed} chunks from the index.")
        st.rerun()

# ── Chat history ──────────────────────────────────────────────────────────────
st.title("Customer Support Multi-Agent Assistant")
st.caption("Natural language over SQL customer data and policy documents · LangGraph · MCP · Chroma")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "chat_meta" not in st.session_state:
    st.session_state.chat_meta = []  # parallel list: {route, rag_sources} per assistant turn

# Re-render past messages with their badges / citations
user_idx = 0
meta_idx = 0
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    else:
        with st.chat_message("assistant"):
            st.markdown(message.content)
            meta = st.session_state.chat_meta[meta_idx] if meta_idx < len(st.session_state.chat_meta) else {}
            route = meta.get("route", "")
            if route:
                st.markdown(route_badge_html(route), unsafe_allow_html=True)
            render_citations(meta.get("rag_sources", []))
            meta_idx += 1

# ── New prompt ────────────────────────────────────────────────────────────────
prompt = st.chat_input("Ask about a customer, ticket, or company policy...")
if prompt:
    st.session_state.chat_history.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = ask(prompt, history=st.session_state.chat_history[:-1])
                answer = result["answer"]
                route = result.get("route", "")
                rag_sources = result.get("rag_sources", [])

                st.markdown(answer)
                if route:
                    st.markdown(route_badge_html(route), unsafe_allow_html=True)
                if route in ("rag", "hybrid"):
                    render_citations(rag_sources)

                st.session_state.chat_history.append(AIMessage(content=answer))
                st.session_state.chat_meta.append({"route": route, "rag_sources": rag_sources})

            except Exception as exc:
                st.error(str(exc))
