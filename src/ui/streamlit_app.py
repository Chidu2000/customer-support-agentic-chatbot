import sys
from pathlib import Path

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.graph import ask
from src.config import settings
from src.ingestion.pdf_processor import ingest_file
from src.vectorstore.chroma_store import add_documents, list_indexed_sources

st.set_page_config(page_title="Customer Support Assistant", page_icon="💬", layout="wide")

st.title("Customer Support Multi-Agent Assistant")
st.caption("Natural language over SQL customer data and policy documents (LangGraph + MCP + Chroma)")

with st.sidebar:
    st.header("Setup")
    st.write(f"Model: `{settings.openai_model}`")
    st.write(f"DB: `{settings.database_url}`")
    # st.write(f"Vector store: `{settings.chroma_persist_dir}`")

    st.divider()
    st.subheader("Upload policy document")
    uploaded = st.file_uploader("PDF or text policy", type=["pdf", "txt", "md"])
    if uploaded is not None:
        policies_dir = Path(settings.policies_dir)
        policies_dir.mkdir(parents=True, exist_ok=True)
        save_path = policies_dir / uploaded.name
        save_path.write_bytes(uploaded.getvalue())
        try:
            chunks = ingest_file(save_path)
            count = add_documents(chunks)
            st.success(f"Ingested {count} chunks from {uploaded.name}")
        except Exception as exc:
            st.error(f"Ingestion failed: {exc}")

    sources = list_indexed_sources()
    st.subheader("Indexed policies")
    if sources:
        for source in sources:
            st.write(f"- {source}")
    else:
        st.info("No policies indexed yet. Upload a file or run `python scripts/ingest_policies.py`.")

    show_debug = st.checkbox("Show agent routing debug", value=False)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

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
                st.markdown(answer)
                if show_debug:
                    st.caption(f"Route: `{result.get('route')}`")
                st.session_state.chat_history.append(AIMessage(content=answer))
            except Exception as exc:
                st.error(str(exc))
