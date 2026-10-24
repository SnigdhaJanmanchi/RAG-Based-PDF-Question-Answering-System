"""AI PDF Assistant — Streamlit UI (Phase 4).

Run (from repo root, with the API already up):

    export API_BASE_URL=http://localhost:8000
    streamlit run frontend/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Allow `streamlit run frontend/streamlit_app.py` without installing a package.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from frontend.api_client import APIClient, APIClientError  # noqa: E402
from frontend.ui_helpers import (  # noqa: E402
    ALL_DOCUMENTS_LABEL,
    EXAMPLE_QUESTIONS,
    assistant_message_from_ask,
    document_choice_labels,
    document_id_for_label,
    file_content_digest,
    format_source_markdown,
    is_insufficient,
    selection_caption,
    should_skip_upload,
)


def _init_state() -> None:
    defaults: dict = {
        "messages": [],
        "processed_uploads": set(),
        "documents": [],
        "docs_loaded": False,
        "selected_label": ALL_DOCUMENTS_LABEL,
        "last_upload_notes": [],
        "backend_status": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _get_client() -> APIClient:
    # Recreate lightly; httpx client is cheap enough for Streamlit scripts.
    return APIClient()


def _refresh_documents(client: APIClient) -> None:
    try:
        st.session_state.documents = client.list_documents()
        st.session_state.docs_loaded = True
        st.session_state.backend_status = "ok"
    except APIClientError as exc:
        st.session_state.backend_status = "error"
        st.session_state.docs_loaded = False
        st.session_state.documents = []
        st.sidebar.error(exc.message)


def _render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})", expanded=False):
        st.caption(
            "Citations show retrieved excerpts used as context. "
            "They do not by themselves verify that the answer is correct."
        )
        for raw in sources:
            from frontend.api_client import AskSource

            source = AskSource(
                document_id=str(raw["document_id"]),
                filename=str(raw["filename"]),
                page=int(raw["page"]),
                chunk_id=str(raw["chunk_id"]),
                excerpt=str(raw["excerpt"]),
                score=float(raw["score"]),
            )
            st.markdown(format_source_markdown(source))
            st.divider()


def main() -> None:
    st.set_page_config(
        page_title="AI PDF Assistant",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _init_state()

    # Minimal contrast-friendly tweaks (native Streamlit components first).
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.25rem; max-width: 1100px; }
        div[data-testid="stSidebar"] { background: #f7f8fa; }
        h1 { letter-spacing: -0.02em; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("AI PDF Assistant")
    st.caption(
        "Ask natural-language questions over your PDFs. "
        "Answers are generated from retrieved document context — always check the cited pages."
    )

    client = _get_client()

    # ----- Sidebar: upload + selection + management -----
    with st.sidebar:
        st.header("Documents")
        st.caption(f"API: `{client.base_url}`")

        if st.button("Refresh library", use_container_width=True):
            _refresh_documents(client)

        if not st.session_state.docs_loaded:
            _refresh_documents(client)

        uploaded = st.file_uploader(
            "Upload PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            help="Files are sent to POST /documents/upload on the FastAPI backend.",
        )

        if uploaded:
            notes: list[str] = []
            progress = st.progress(0.0, text="Preparing upload…")
            total = len(uploaded)
            for index, file in enumerate(uploaded, start=1):
                content = file.getvalue()
                digest = file_content_digest(content)
                if should_skip_upload(digest, st.session_state.processed_uploads):
                    notes.append(f"Skipped (already uploaded this session): {file.name}")
                    progress.progress(index / total, text=f"Checked {file.name}")
                    continue
                progress.progress((index - 0.5) / total, text=f"Indexing {file.name}…")
                try:
                    result = client.upload_pdf(
                        filename=file.name or "upload.pdf",
                        content=content,
                        content_type=file.type or "application/pdf",
                    )
                    st.session_state.processed_uploads.add(digest)
                    if result.duplicate:
                        notes.append(
                            f"Already indexed: {result.filename} "
                            f"({result.page_count} pages, {result.chunk_count} chunks)"
                        )
                    else:
                        notes.append(
                            f"Indexed: {result.filename} — "
                            f"{result.page_count} pages, {result.chunk_count} chunks"
                        )
                except APIClientError as exc:
                    notes.append(f"Failed: {file.name} — {exc.message}")
                progress.progress(index / total, text=f"Finished {file.name}")
            progress.empty()
            st.session_state.last_upload_notes = notes
            _refresh_documents(client)

        for note in st.session_state.last_upload_notes:
            if note.startswith("Failed:"):
                st.error(note)
            elif note.startswith("Skipped"):
                st.info(note)
            else:
                st.success(note)

        st.subheader("Ask scope")
        documents = st.session_state.documents
        if not documents:
            st.warning("Library is empty. Upload a text-based PDF to begin.")
            st.session_state.selected_label = ALL_DOCUMENTS_LABEL
            selected_doc_id = None
        else:
            labels = document_choice_labels(documents)
            # Keep selection stable across refreshes when possible.
            current = st.session_state.selected_label
            if current not in labels:
                current = ALL_DOCUMENTS_LABEL
            choice = st.radio(
                "Document selection",
                options=labels,
                index=labels.index(current),
                label_visibility="collapsed",
            )
            st.session_state.selected_label = choice
            selected_doc_id = document_id_for_label(choice, documents)

        st.caption(selection_caption(selected_doc_id, documents))

        st.divider()
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.caption(
            "OpenAI key stays on the API server (.env). "
            "This UI never sends or displays API keys."
        )

    # ----- Main: chat -----
    selected_doc_id = (
        None
        if not st.session_state.documents
        else document_id_for_label(st.session_state.selected_label, st.session_state.documents)
    )

    if not st.session_state.messages:
        st.info(
            "**Get started**\n\n"
            "1. Upload one or more PDFs in the sidebar\n"
            "2. Choose a single document or search all\n"
            "3. Ask a question below\n\n"
            "Example questions:\n"
            + "\n".join(f"- {q}" for q in EXAMPLE_QUESTIONS)
        )

    for message in st.session_state.messages:
        role = message.get("role", "assistant")
        with st.chat_message(role):
            if role == "assistant" and is_insufficient(str(message.get("status", ""))):
                st.warning(message.get("content", ""))
            else:
                st.markdown(message.get("content", ""))
            if role == "assistant":
                _render_sources(list(message.get("sources") or []))

    prompt = st.chat_input(
        "Ask a question about your PDFs…",
        disabled=not st.session_state.documents,
    )
    if prompt:
        # Append user turn once; chat_input is None on subsequent reruns → no repeat calls.
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving context and generating an answer…"):
                try:
                    result = client.ask(
                        prompt,
                        document_id=selected_doc_id,
                        top_k=None,
                    )
                    msg = assistant_message_from_ask(result)
                    st.session_state.messages.append(msg)
                    if is_insufficient(result.status):
                        st.warning(result.answer)
                    else:
                        st.markdown(result.answer)
                    _render_sources(
                        [
                            {
                                "document_id": s.document_id,
                                "filename": s.filename,
                                "page": s.page,
                                "chunk_id": s.chunk_id,
                                "excerpt": s.excerpt,
                                "score": s.score,
                            }
                            for s in result.sources
                        ]
                    )
                except APIClientError as exc:
                    err_text = exc.message
                    st.error(err_text)
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": err_text,
                            "status": "error",
                            "sources": [],
                            "documents": [],
                        }
                    )


if __name__ == "__main__":
    main()
