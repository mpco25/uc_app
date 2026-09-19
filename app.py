# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "streamlit",
#   "faiss-cpu",
#   "httpx",
#   "numpy",
# ]
# ///

import json
from pathlib import Path

import faiss
import httpx
import streamlit as st

from llm import (
    ask_model,
    build_context,
    get_models,
)

from search.exact import exact_search
from search.hybrid import hybrid_search
from search.semantic import semantic_search

from ui import (
    render_exact_summary,
    render_table_details,
    render_table_overview,
)

from config import (
    INDEX_FILE,
    METADATA_FILE,
    OLLAMA_BASE_URL,
    EMBEDDING_MODEL,
    REQUEST_TIMEOUT,
    PREFERRED_MODEL,
    DEFAULT_CANDIDATES,
)

@st.cache_resource
def load_index():
    if not INDEX_FILE.exists():
        raise FileNotFoundError(
            f"FAISS index not found: {INDEX_FILE.resolve()}"
        )

    return faiss.read_index(str(INDEX_FILE))


@st.cache_resource
def load_metadata():
    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {METADATA_FILE.resolve()}"
        )

    with METADATA_FILE.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    if not isinstance(metadata, list):
        raise ValueError(
            "uc_metadata.json must contain a JSON array."
        )

    return metadata


st.set_page_config(
    page_title="Unity Catalog Assistant",
    layout="wide",
)

st.title("Unity Catalog Metadata Assistant")

try:
    index = load_index()
    metadata = load_metadata()
except Exception as error:
    st.error(str(error))
    st.stop()

if index.ntotal != len(metadata):
    st.error(
        f"The FAISS index contains {index.ntotal} vectors, but the "
        f"metadata file contains {len(metadata)} records. Rebuild the "
        "index so both files match."
    )
    st.stop()


with st.sidebar:
    st.header("Settings")

    try:
        models = get_models()
    except Exception as error:
        st.error(
            f"Could not retrieve models from Ollama: {error}"
        )
        models = []

    if models:
        default_model_index = (
            models.index(PREFERRED_MODEL)
            if PREFERRED_MODEL in models
            else 0
        )

        selected_model = st.selectbox(
            "LLM Model",
            models,
            index=default_model_index,
        )
    else:
        selected_model = None
        st.warning("No Ollama generation models were found.")

    search_mode = st.radio(
        "Search Mode",
        ["Hybrid", "Exact", "Semantic"],
        help=(
            "Hybrid combines keyword matching with meaning-based search. "
            "Exact searches the entered text literally across all metadata. "
            "Semantic searches by meaning using embeddings."
        ),
    )

    candidates = st.slider(
        "Tables considered",
        min_value=5,
        max_value=100,
        value=DEFAULT_CANDIDATES,
        step=5,
        disabled=search_mode == "Exact",
        help=(
            "How many potentially relevant tables are reviewed before "
            "an answer is generated. Exact mode searches all metadata, "
            "so this setting does not apply to Exact mode."
        ),
    )

    show_context = st.checkbox(
        "Show Retrieved Context",
        value=False,
        disabled=search_mode == "Exact",
    )

    st.caption(f"Indexed tables: {index.ntotal}")

    st.link_button(
        "❓ Open Help",
        "http://localhost:8501/Help",
    )

input_label = (
    "Exact search text"
    if search_mode == "Exact"
    else "Question"
)

input_placeholder = (
    "For example: cell"
    if search_mode == "Exact"
    else "For example: Where are purchase orders stored?"
)

search_text = st.text_area(
    input_label,
    height=120,
    placeholder=input_placeholder,
)

button_label = "Search" if search_mode == "Exact" else "Ask"

if st.button(button_label, type="primary"):
    cleaned_input = search_text.strip()

    if not cleaned_input:
        st.warning("Please enter search text.")
        st.stop()

    st.info(f"Retrieval method used: {search_mode}")

    try:
        if search_mode == "Exact":
            with st.spinner("Searching all metadata..."):
                results = exact_search(
                    metadata=metadata,
                    search_text=cleaned_input,
                )

            if not results:
                st.warning(
                    f'No exact matches found for "{cleaned_input}".'
                )
                st.stop()

            render_exact_summary(results, cleaned_input)
            render_table_overview(results)
            render_table_details(results)
            st.stop()

        if not selected_model:
            st.warning(
                "Please install or select an Ollama generation model."
            )
            st.stop()

        with st.spinner("Searching metadata..."):
            if search_mode == "Semantic":
                results = semantic_search(
                    index=index,
                    metadata=metadata,
                    question=cleaned_input,
                    top_n=candidates,
                    ollama_base_url=OLLAMA_BASE_URL,
                    embedding_model=EMBEDDING_MODEL,
                    request_timeout=REQUEST_TIMEOUT,
                )
            else:
                results = hybrid_search(
                    index=index,
                    metadata=metadata,
                    question=cleaned_input,
                    top_n=candidates,
                    ollama_base_url=OLLAMA_BASE_URL,
                    embedding_model=EMBEDDING_MODEL,
                    request_timeout=REQUEST_TIMEOUT,
                )

        if not results:
            st.warning("No metadata records were retrieved.")
            st.stop()

        render_table_overview(results)
        render_table_details(results)

        if show_context:
            st.subheader("Retrieved Context")
            st.code(
                build_context(results),
                language="json",
            )

        with st.spinner(f"Asking {selected_model}..."):
            answer = ask_model(
                selected_model,
                cleaned_input,
                results,
            )

        st.subheader("Answer")
        st.markdown(answer)

    except httpx.ConnectError:
        st.error(
            f"Could not connect to Ollama at {OLLAMA_BASE_URL}. "
            "Check that Ollama is running."
        )
    except httpx.HTTPStatusError as error:
        st.error(
            f"Ollama returned HTTP {error.response.status_code}."
        )
        st.code(error.response.text)
    except Exception as error:
        st.error(f"Application error: {error}")
