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
import numpy as np
import streamlit as st


INDEX_FILE = Path("data/uc_metadata.index")
METADATA_FILE = Path("data/uc_metadata.json")

OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text:latest"

REQUEST_TIMEOUT = 600
TOP_N = 25
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

    models = [
        model["name"]
        for model in response.json().get("models", [])
        if "embed" not in model.get("name", "").lower()
    ]

    return sorted(models)


def create_embedding(text: str):
    response = httpx.post(
        f"{OLLAMA_BASE_URL}/api/embed",
        json={
            "model": EMBEDDING_MODEL,
            "input": text,
            "truncate": True,
            "keep_alive": "10m",
        },
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()

    embeddings = response.json().get("embeddings")
    if not embeddings:
        raise RuntimeError("Ollama returned no embedding.")

    vector = np.asarray(
        [embeddings[0]],
        dtype=np.float32,
    )

    faiss.normalize_L2(vector)
    return vector


def semantic_search(
    index,
    metadata,
    question,
    top_n=TOP_N,
):
    vector = create_embedding(question)

    scores, positions = index.search(
        vector,
        min(top_n, index.ntotal),
    )

    results = []

    for score, position in zip(
        scores[0],
        positions[0],
        strict=True,
    ):
        if position < 0:
            continue

        if position >= len(metadata):
            raise IndexError(
                f"FAISS returned position {position}, but metadata "
                f"contains only {len(metadata)} records."
            )

        results.append(
            {
                "similarity": float(score),
                "metadata": metadata[position],
            }
        )

    return results


def build_context(results):
    payload = []

    for result in results:
        row = result["metadata"]

        payload.append(
            {
                "similarity": round(
                    result["similarity"],
                    4,
                ),
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


def ask_model(
    model,
    question,
    results,
):
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
                    "content": """
You are a Databricks Unity Catalog metadata assistant.

Use only the metadata provided.
Do not invent catalogs, schemas, tables, columns, or relationships.
Similarity scores are retrieval hints, not proof.
Verify claims using table names, table descriptions, column names,
and column descriptions.
If there is insufficient evidence, say so explicitly.
Do not claim that semantic search results are exhaustive.
Keep the answer concise.
""".strip(),
                },
                {
                    "role": "user",
                    "content": f"""
Question:
{question}

Metadata:
{context}
""".strip(),
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

    table_rows = []

    for result in results:
        row = result["metadata"]
        description = row.get("table_comment") or "(No description)"

        table_rows.append(
            {
                "Similarity": round(
                    result["similarity"],
                    4,
                ),
                "Catalog": row.get("catalog"),
                "Schema": row.get("schema"),
                "Table": row.get("table"),
                "Description": description,
                "Columns": len(row.get("columns", [])),
            }
        )

    st.dataframe(
        table_rows,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Similarity": st.column_config.NumberColumn(
                "Similarity",
                format="%.4f",
            ),
            "Description": st.column_config.TextColumn(
                "Description",
                width="large",
            ),
        },
    )


def render_table_details(results):
    st.subheader("Retrieved Table Details")

    for result in results:
        row = result["metadata"]

        catalog = row.get("catalog") or ""
        schema = row.get("schema") or ""
        table = row.get("table") or ""
        description = row.get("table_comment") or "(No description)"
        columns = row.get("columns", [])

        title = (
            f"{result['similarity']:.4f} | "
            f"{catalog}.{schema}.{table}"
        )

        with st.expander(title):
            st.markdown(f"**Description:** {description}")
            st.markdown(f"**Total columns:** {len(columns)}")

            if columns:
                st.markdown(
                    f"**First {min(SAMPLE_COLUMNS, len(columns))} columns:**"
                )

                column_rows = []

                for column in columns[:SAMPLE_COLUMNS]:
                    column_rows.append(
                        {
                            "Column": column.get("name"),
                            "Description": (
                                column.get("comment")
                                or "(No description)"
                            ),
                        }
                    )

                st.dataframe(
                    column_rows,
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No columns are available in the metadata record.")


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
        st.error(f"Could not retrieve models from Ollama: {error}")
        models = []

    if not models:
        st.warning("No Ollama generation models were found.")
        selected_model = None
    else:
        preferred_model = "qwen3.8:latest"
        default_index = (
            models.index(preferred_model)
            if preferred_model in models
            else 0
        )

        selected_model = st.selectbox(
            "LLM Model",
            models,
            index=default_index,
        )

    search_mode = st.radio(
        "Search Mode",
        [
            "Semantic",
            "Auto",
            "Exact",
            "Hybrid",
        ],
        help=(
            "The current version implements semantic retrieval. "
            "Auto, Exact, and Hybrid are UI placeholders."
        ),
    )

    selected_top_n = st.slider(
        "Tables considered",
        min_value=5,
        max_value=100,
        value=25,
        step=5,
        help=(
            "How many potentially relevant tables are reviewed "
            "before generating an answer. Higher values may "
            "find more matches but take longer."
        ),
    )

    show_context = st.checkbox(
        "Show Retrieved Context",
        value=False,
    )

    st.caption(
        f"Indexed tables: {index.ntotal}"
    )

question = st.text_area(
    "Question",
    height=120,
    placeholder="Where are purchase orders stored?",
)

if st.button("Ask", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    if not selected_model:
        st.warning("Please install or select an Ollama generation model.")
        st.stop()

    if search_mode != "Semantic":
        st.info(
            f"{search_mode} mode is not implemented yet. "
            "This request is using semantic retrieval."
        )

    try:
        with st.spinner("Searching metadata..."):
            results = semantic_search(
                index=index,
                metadata=metadata,
                question=question,
                top_n=selected_top_n,
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
                question,
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
