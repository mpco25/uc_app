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

from search.exact import exact_search
from search.hybrid import hybrid_search
from search.semantic import semantic_search


INDEX_FILE = Path("data/uc_metadata.index")
METADATA_FILE = Path("data/uc_metadata.json")

OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text:latest"
PREFERRED_MODEL = "qwen3.8:latest"

REQUEST_TIMEOUT = 600
DEFAULT_CANDIDATES = 25
SAMPLE_COLUMNS = 10


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


@st.cache_data(ttl=30)
def get_models():
    response = httpx.get(
        f"{OLLAMA_BASE_URL}/api/tags",
        timeout=30,
    )
    response.raise_for_status()

    return sorted(
        model["name"]
        for model in response.json().get("models", [])
        if "embed" not in model.get("name", "").lower()
    )


def build_context(results):
    payload = []

    for result in results:
        row = result["metadata"]

        payload.append(
            {
                "source": result.get("source"),
                "similarity": (
                    None
                    if result.get("similarity") is None
                    else round(result["similarity"], 4)
                ),
                "keyword_score": result.get("lexical_score", 0),
                "matched_terms": result.get("matched_terms", []),
                "catalog": row.get("catalog"),
                "schema": row.get("schema"),
                "table": row.get("table"),
                "type": row.get("type"),
                "table_comment": row.get("table_comment"),
                "columns": row.get("columns", []),
            }
        )

    return json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
    )


def ask_model(model, question, results):
    context = build_context(results)

    response = httpx.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": model,
            "stream": False,
            "keep_alive": "10m",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a Databricks Unity Catalog metadata "
                        "assistant. Use only the supplied metadata. Do not "
                        "invent catalogs, schemas, tables, columns, or "
                        "relationships. Verify claims using names and "
                        "descriptions. If evidence is insufficient, say so. "
                        "Do not claim that retrieved results are exhaustive. "
                        "Keep the answer concise."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question:\n{question}\n\n"
                        f"Metadata:\n{context}"
                    ),
                },
            ],
            "options": {
                "temperature": 0,
            },
        },
        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()

    answer = response.json().get("message", {}).get("content")

    if not answer:
        raise RuntimeError("Ollama returned no answer.")

    return answer.strip()


def render_table_overview(results):
    st.subheader("Retrieved Tables")

    rows = []

    for result in results:
        row = result["metadata"]

        rows.append(
            {
                "Source": result.get("source"),
                "Similarity": result.get("similarity"),
                "Keyword score": result.get("lexical_score", 0),
                "Catalog": row.get("catalog"),
                "Schema": row.get("schema"),
                "Table": row.get("table"),
                "Description": (
                    row.get("table_comment")
                    or "(No description)"
                ),
                "Columns": len(row.get("columns", [])),
            }
        )

    st.dataframe(
        rows,
        use_container_width=True,
        hide_index=True,
    )


def render_table_details(results):
    st.subheader("Retrieved Table Details")

    for result in results:
        row = result["metadata"]
        similarity = result.get("similarity")
        score = (
            "Exact match"
            if similarity is None
            else f"{similarity:.4f}"
        )
        title = (
            f"{score} | "
            f"{row.get('catalog')}."
            f"{row.get('schema')}."
            f"{row.get('table')}"
        )

        with st.expander(title):
            st.markdown(
                f"**Description:** "
                f"{row.get('table_comment') or '(No description)'}"
            )

            match_locations = result.get("match_locations", [])

            if match_locations:
                st.markdown("**Matched in:**")
                st.dataframe(
                    [
                        {
                            "Location": match.get("location"),
                            "Column": match.get("column"),
                            "Value": match.get("value"),
                        }
                        for match in match_locations
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

            columns = row.get("columns", [])
            st.markdown(f"**Total columns:** {len(columns)}")

            if columns:
                st.markdown(
                    f"**First {min(SAMPLE_COLUMNS, len(columns))} "
                    "columns:**"
                )
                st.dataframe(
                    [
                        {
                            "Column": column.get("name"),
                            "Description": (
                                column.get("comment")
                                or "(No description)"
                            ),
                        }
                        for column in columns[:SAMPLE_COLUMNS]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )


def render_exact_summary(results, search_text):
    st.success(
        f'{len(results)} tables contain the exact text "{search_text}".'
    )

    match_count = sum(
        len(result.get("match_locations", []))
        for result in results
    )

    st.caption(
        f"Total matching metadata locations: {match_count}. "
        "Exact mode searches catalog, schema, table, and column names "
        "and descriptions."
    )


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
