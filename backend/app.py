"""
app.py – Streamlit UI for the RAG Chatbot

Features
--------
- URL ingestion with progress bar
- Toggle between LangChain and LlamaIndex backends
- Chat interface with streaming answers
- Source citations shown under each answer
- Knowledge base stats panel
"""
from __future__ import annotations

import json
import time

import requests
import streamlit as st

# ── Config ──────────────────────────────────────────────────────────────────
API_URL = "http://localhost:8000"   # override via st.secrets or env in prod

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styles ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.source-card {
    background: #1e1e2e;
    border: 1px solid #313244;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 6px 0;
    font-size: 0.85rem;
    color: #cdd6f4;
}
.source-card a { color: #89b4fa; }
.framework-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-bottom: 6px;
}
.badge-langchain  { background: #1c7c54; color: #fff; }
.badge-llamaindex { background: #7c3d1c; color: #fff; }
</style>
""", unsafe_allow_html=True)

# ── Session state ────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []          # {"role", "content", "sources"}
if "framework" not in st.session_state:
    st.session_state.framework = "langchain"


# ── Helpers ──────────────────────────────────────────────────────────────────

def api_ingest(url: str, framework: str, raw_text: str = "", title: str = "") -> dict:
    resp = requests.post(
        f"{API_URL}/ingest",
        json={"url": url, "framework": framework, "raw_text": raw_text, "title": title},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def api_collections() -> dict:
    try:
        return requests.get(f"{API_URL}/collections", timeout=10).json()
    except Exception:
        return {}


def api_sources(framework: str) -> list:
    try:
        return requests.get(f"{API_URL}/collections/{framework}/sources", timeout=10).json()
    except Exception:
        return []


def api_delete(framework: str) -> None:
    requests.delete(f"{API_URL}/collections/{framework}", timeout=10)


def api_delete_url(framework: str, collection_name: str) -> None:
    requests.delete(f"{API_URL}/collections/{framework}/url/{collection_name}", timeout=10)


def stream_chat(question: str, framework: str):
    """
    Yields (token: str, sources: list | None) tuples.
    sources is None for every token event; it is a list on the final event.
    """
    with requests.post(
        f"{API_URL}/chat",
        json={"question": question, "framework": framework},
        stream=True,
        timeout=120,
    ) as resp:
        resp.raise_for_status()
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
            if not line.startswith("data: "):
                continue
            payload = json.loads(line[6:])
            if payload["type"] == "token":
                yield payload["data"], None
            elif payload["type"] == "sources":
                yield "", payload["data"]
            elif payload["type"] == "done":
                return
            elif payload["type"] == "error":
                raise RuntimeError(payload["data"])


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Settings")

    st.subheader("Framework")
    framework = st.radio(
        "RAG backend",
        options=["langchain", "llamaindex"],
        format_func=lambda x: "🦜 LangChain" if x == "langchain" else "🦙 LlamaIndex",
        key="framework",
        help=(
            "LangChain: RecursiveCharacterTextSplitter + RetrievalQA chain\n"
            "LlamaIndex: SentenceSplitter + VectorStoreIndex query engine"
        ),
    )

    st.divider()

    st.subheader("📥 Add to Knowledge Base")

    ingest_mode = st.radio(
        "Input method",
        ["🔗 Scrape URL", "📋 Paste text"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if ingest_mode == "🔗 Scrape URL":
        with st.form("ingest_form_url"):
            url_input = st.text_input("URL to scrape", placeholder="https://example.com/article")
            st.caption("Uses a real browser — most JS-rendered pages work. Paste text for paywalled sites.")
            submitted = st.form_submit_button("Process URL", use_container_width=True)

        if submitted and url_input:
            progress = st.progress(0, text="Fetching page…")
            try:
                progress.progress(20, text="Scraping & cleaning HTML…")
                time.sleep(0.3)
                progress.progress(50, text="Embedding chunks…")
                result = api_ingest(url_input, framework)
                progress.progress(90, text="Storing in Qdrant…")
                time.sleep(0.2)
                progress.progress(100, text="Done!")
                st.success(
                    f"✅ **{result['title']}**\n\n"
                    f"{result['chunks_stored']} chunks stored via **{result['framework']}**"
                )
            except requests.HTTPError as e:
                st.error(f"API error: {e.response.json().get('detail', str(e))}")
            except Exception as e:
                st.error(f"Error: {e}")

    else:  # Paste text mode
        with st.form("ingest_form_text"):
            paste_url = st.text_input(
                "Source URL (for citation)",
                placeholder="https://medium.com/your-article",
            )
            paste_title = st.text_input("Title", placeholder="Article title")
            paste_text = st.text_area(
                "Paste article text here",
                height=200,
                placeholder="Copy and paste the full article text…",
            )
            submitted_text = st.form_submit_button("Process Text", use_container_width=True)

        if submitted_text and paste_text and paste_url:
            progress = st.progress(0, text="Processing text…")
            try:
                progress.progress(40, text="Embedding chunks…")
                result = api_ingest(
                    url=paste_url,
                    framework=framework,
                    raw_text=paste_text,
                    title=paste_title or paste_url,
                )
                progress.progress(100, text="Done!")
                st.success(
                    f"✅ **{result['title']}**\n\n"
                    f"{result['chunks_stored']} chunks stored via **{result['framework']}**"
                )
            except requests.HTTPError as e:
                st.error(f"API error: {e.response.json().get('detail', str(e))}")
            except Exception as e:
                st.error(f"Error: {e}")

    st.divider()

    st.subheader("📊 Knowledge Base")
    if st.button("Refresh stats", use_container_width=True):
        st.session_state["kb_stats"] = api_collections()
        st.session_state["kb_sources_llamaindex"] = api_sources("llamaindex")
        st.session_state["kb_sources_langchain"] = api_sources("langchain")

    stats = st.session_state.get("kb_stats", api_collections())
    for fw, count in stats.items():
        label = "🦜 LangChain" if fw == "langchain" else "🦙 LlamaIndex"
        st.metric(label=label, value=f"{count} chunks")

    # Per-URL source list for the active framework
    src_key = f"kb_sources_{framework}"
    sources_list = st.session_state.get(src_key, api_sources(framework))
    if sources_list:
        st.caption(f"Ingested URLs ({framework}):")
        for src in sources_list:
            col_title, col_del = st.columns([4, 1])
            with col_title:
                st.markdown(
                    f"[{src['title'][:40] or src['source']}]({src['source']}) "
                    f"— {src['chunks']} chunks",
                    help=src["source"],
                )
            with col_del:
                if st.button("✕", key=f"del_{src['collection']}", help="Remove this URL"):
                    api_delete_url(framework, src["collection"])
                    st.session_state.pop(src_key, None)
                    st.rerun()

    st.divider()

    st.subheader("🗑️ Clear Knowledge Base")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("LangChain", use_container_width=True):
            api_delete("langchain")
            st.success("LangChain KB cleared")
    with col2:
        if st.button("LlamaIndex", use_container_width=True):
            api_delete("llamaindex")
            st.success("LlamaIndex KB cleared")

    if st.button("Clear chat history", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── Main chat panel ──────────────────────────────────────────────────────────

st.title("🔍 RAG Chatbot")
badge_class = "badge-langchain" if framework == "langchain" else "badge-llamaindex"
badge_label = "🦜 LangChain" if framework == "langchain" else "🦙 LlamaIndex"
st.markdown(
    f'<span class="framework-badge {badge_class}">{badge_label}</span>',
    unsafe_allow_html=True,
)
st.caption("Add URLs in the sidebar, then ask questions about them below.")

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📚 {len(msg['sources'])} source(s) used", expanded=False):
                for src in msg["sources"]:
                    st.markdown(
                        f'<div class="source-card">'
                        f'<a href="{src["source"]}" target="_blank">{src["title"] or src["source"]}</a>'
                        f"<br/><small>{src['content'][:300]}…</small>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

# Chat input
if question := st.chat_input("Ask a question about your knowledge base…"):
    # Show user bubble
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Stream assistant response
    with st.chat_message("assistant"):
        answer_placeholder = st.empty()
        full_answer = ""
        sources: list = []

        try:
            for token, src in stream_chat(question, framework):
                if src is not None:
                    sources = src
                else:
                    full_answer += token
                    answer_placeholder.markdown(full_answer + "▌")

            answer_placeholder.markdown(full_answer)

            if sources:
                with st.expander(f"📚 {len(sources)} source(s) used", expanded=False):
                    for src in sources:
                        st.markdown(
                            f'<div class="source-card">'
                            f'<a href="{src["source"]}" target="_blank">{src["title"] or src["source"]}</a>'
                            f"<br/><small>{src['content'][:300]}…</small>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
        except Exception as e:
            full_answer = f"❌ Error: {e}"
            answer_placeholder.error(full_answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": full_answer, "sources": sources}
        )