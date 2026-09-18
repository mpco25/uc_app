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
import re
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
DEFAULT_CANDIDATES = 25
SAMPLE_COLUMNS = 10

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "contain",
    "contains", "containing", "data", "do", "does", "every", "find", "for",
    "from", "has", "have", "how", "i", "in", "information", "is", "it",
    "list", "me", "of", "on", "or", "show", "stored", "table", "tables",
    "the", "there", "to", "what", "where", "which", "with",
}


@st.cache_resource
def load_index():
    if not INDEX_FILE.exists():
        raise FileNotFoundError(f"FAISS index not found: {INDEX_FILE.resolve()}")
    return faiss.read_index(str(INDEX_FILE))


@st.cache_resource
def load_metadata():
    if not METADATA_FILE.exists():
        raise FileNotFoundError(f"Metadata file not found: {METADATA_FILE.resolve()}")
    with METADATA_FILE.open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    if not isinstance(metadata, list):
        raise ValueError("uc_metadata.json must contain a JSON array.")
    return metadata


@st.cache_data(ttl=30)
def get_models():
    response = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=30)
    response.raise_for_status()
    return sorted(
        model["name"]
        for model in response.json().get("models", [])
        if "embed" not in model.get("name", "").lower()
    )


def create_embedding(text):
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
    vector = np.asarray([embeddings[0]], dtype=np.float32)
    faiss.normalize_L2(vector)
    return vector


def semantic_search(index, metadata, question, top_n):
    vector = create_embedding(question)
    scores, positions = index.search(vector, min(top_n, index.ntotal))
    results = []
    for rank, (score, position) in enumerate(
        zip(scores[0], positions[0], strict=True), start=1
    ):
        if position < 0:
            continue
        results.append(
            {
                "position": int(position),
                "semantic_rank": rank,
                "similarity": float(score),
                "lexical_score": 0,
                "metadata": metadata[position],
            }
        )
    return results


def tokenize(text):
    return [
        token
        for token in re.findall(r"[a-z0-9_]+", text.casefold())
        if len(token) > 1 and token not in STOPWORDS
    ]


def searchable_text(record):
    parts = [
        record.get("catalog") or "",
        record.get("catalog_comment") or "",
        record.get("schema") or "",
        record.get("schema_comment") or "",
        record.get("table") or "",
        record.get("table_comment") or "",
    ]
    for column in record.get("columns", []):
        parts.extend([column.get("name") or "", column.get("comment") or ""])
    return " ".join(parts).casefold()


def lexical_search(metadata, question):
    terms = tokenize(question)
    if not terms:
        return []

    results = []
    for position, record in enumerate(metadata):
        text = searchable_text(record)
        score = 0
        matched_terms = []
        table_name = (record.get("table") or "").casefold()

        for term in terms:
            occurrences = text.count(term)
            if occurrences:
                matched_terms.append(term)
                score += occurrences
                if term in table_name:
                    score += 5
                for column in record.get("columns", []):
                    if term in (column.get("name") or "").casefold():
                        score += 3

        if score:
            results.append(
                {
                    "position": position,
                    "semantic_rank": None,
                    "similarity": None,
                    "lexical_score": score,
                    "matched_terms": matched_terms,
                    "metadata": record,
                }
            )

    return sorted(results, key=lambda item: item["lexical_score"], reverse=True)


def hybrid_search(index, metadata, question, top_n):
    semantic = semantic_search(index, metadata, question, max(top_n * 2, 50))
    lexical = lexical_search(metadata, question)
    merged = {}

    for rank, result in enumerate(semantic, start=1):
        item = dict(result)
        item["hybrid_score"] = 1.0 / (60 + rank)
        merged[item["position"]] = item

    for rank, result in enumerate(lexical, start=1):
        position = result["position"]
        reciprocal = 1.0 / (60 + rank)
        if position in merged:
            merged[position]["hybrid_score"] += reciprocal
            merged[position]["lexical_score"] = result["lexical_score"]
            merged[position]["matched_terms"] = result.get("matched_terms", [])
        else:
            item = dict(result)
            item["hybrid_score"] = reciprocal
            merged[position] = item

    return sorted(
        merged.values(), key=lambda item: item["hybrid_score"], reverse=True
    )[:top_n]


def extract_exact_request(question):
    normalized = " ".join(question.strip().split())
    quoted = re.search(r"['\"]([^'\"]+)['\"]", normalized)
    quoted_term = quoted.group(1).strip() if quoted else None

    patterns = [
        ("column_name", r"columns?.*?names?.*?(?:contains?|containing|includes?|including|mentions?|mentioning|starts? with|ends? with)\s+(.+?)(?:[?.]|$)"),
        ("column_name", r"columns?.*?(?:contains?|containing|includes?|including|mentions?|mentioning|starts? with|ends? with)\s+(.+?)(?:[?.]|$)"),
        ("column_comment", r"columns?.*?(?:descriptions?|comments?).*?(?:contains?|containing|includes?|including|mentions?|mentioning)\s+(.+?)(?:[?.]|$)"),
        ("table_name", r"tables?.*?names?.*?(?:contains?|containing|includes?|including|mentions?|mentioning|starts? with|ends? with)\s+(.+?)(?:[?.]|$)"),
        ("table_name", r"tables?.*?(?:contains?|containing|includes?|including|mentions?|mentioning|starts? with|ends? with)\s+(.+?)(?:[?.]|$)"),
        ("table_comment", r"tables?.*?(?:descriptions?|comments?).*?(?:contains?|containing|includes?|including|mentions?|mentioning)\s+(.+?)(?:[?.]|$)"),
    ]

    lower = normalized.casefold()
    for field, pattern in patterns:
        match = re.search(pattern, lower)
        if match:
            term = quoted_term or match.group(1).strip(" '`\"")
            if term:
                return field, term
    return None


def exact_search(metadata, field, term):
    needle = term.casefold()
    results = []
    for record in metadata:
        if field in {"table_name", "table_comment"}:
            value = record.get("table") if field == "table_name" else record.get("table_comment")
            if needle in (value or "").casefold():
                results.append(
                    {
                        "Catalog": record.get("catalog"),
                        "Schema": record.get("schema"),
                        "Table": record.get("table"),
                        "Column": None,
                        "Description": record.get("table_comment") or "(No description)",
                    }
                )
            continue

        for column in record.get("columns", []):
            value = column.get("name") if field == "column_name" else column.get("comment")
            if needle in (value or "").casefold():
                results.append(
                    {
                        "Catalog": record.get("catalog"),
                        "Schema": record.get("schema"),
                        "Table": record.get("table"),
                        "Column": column.get("name"),
                        "Description": column.get("comment") or "(No description)",
                    }
                )
    return results


def build_context(results):
    payload = []
    for result in results:
        row = result["metadata"]
        payload.append(
            {
                "similarity": None if result.get("similarity") is None else round(result["similarity"], 4),
                "lexical_score": result.get("lexical_score", 0),
                "catalog": row.get("catalog"),
                "schema": row.get("schema"),
                "table": row.get("table"),
                "type": row.get("type"),
                "table_comment": row.get("table_comment"),
                "columns": row.get("columns", []),
            }
        )
    return json.dumps(payload, ensure_ascii=False, indent=2)


def ask_model(model, question, results):
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
                        "You are a Databricks Unity Catalog metadata assistant. "
                        "Use only the supplied metadata. Do not invent objects or relationships. "
                        "Verify claims using names and descriptions. If evidence is insufficient, "
                        "say so. Do not claim the result is exhaustive. Keep the answer concise."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Question:\n{question}\n\nMetadata:\n{build_context(results)}",
                },
            ],
            "options": {"temperature": 0},
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
                "Similarity": result.get("similarity"),
                "Keyword score": result.get("lexical_score", 0),
                "Catalog": row.get("catalog"),
                "Schema": row.get("schema"),
                "Table": row.get("table"),
                "Description": row.get("table_comment") or "(No description)",
                "Columns": len(row.get("columns", [])),
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)


def render_table_details(results):
    st.subheader("Retrieved Table Details")
    for result in results:
        row = result["metadata"]
        similarity = result.get("similarity")
        score = "n/a" if similarity is None else f"{similarity:.4f}"
        title = f"{score} | {row.get('catalog')}.{row.get('schema')}.{row.get('table')}"
        with st.expander(title):
            st.markdown(f"**Description:** {row.get('table_comment') or '(No description)'}")
            columns = row.get("columns", [])
            st.markdown(f"**Total columns:** {len(columns)}")
            if columns:
                st.dataframe(
                    [
                        {
                            "Column": column.get("name"),
                            "Description": column.get("comment") or "(No description)",
                        }
                        for column in columns[:SAMPLE_COLUMNS]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )


st.set_page_config(page_title="Unity Catalog Assistant", layout="wide")
st.title("Unity Catalog Metadata Assistant")

try:
    index = load_index()
    metadata = load_metadata()
except Exception as error:
    st.error(str(error))
    st.stop()

if index.ntotal != len(metadata):
    st.error(
        f"Index contains {index.ntotal} vectors, but metadata contains "
        f"{len(metadata)} records. Rebuild the index."
    )
    st.stop()

with st.sidebar:
    st.header("Settings")
    try:
        models = get_models()
    except Exception as error:
        st.error(f"Could not retrieve models from Ollama: {error}")
        models = []

    selected_model = st.selectbox("LLM Model", models) if models else None
    search_mode = st.radio(
        "Search Mode",
        ["Hybrid", "Exact", "Semantic"],
        help=(
            "Hybrid combines exact keyword matching with meaning-based search. "
            "Exact performs literal keyword matching. "
            "Semantic searches by meaning using embeddings."
        ),
    )
    candidates = st.slider(
        "Tables considered",
        min_value=5,
        max_value=100,
        value=DEFAULT_CANDIDATES,
        step=5,
        help=(
            "How many potentially relevant tables are reviewed before an answer is generated. "
            "Higher values may improve coverage but can make answers slower."
        ),
    )
    show_context = st.checkbox("Show Retrieved Context", value=False)
    st.caption(f"Indexed tables: {index.ntotal}")

question = st.text_area(
    "Question",
    height=120,
    placeholder="Where are purchase orders stored?",
)

if st.button("Ask", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    effective_mode = search_mode
    exact_request = (
        extract_exact_request(question)
        if search_mode == "Exact"
        else None
    )

    st.info(f"Retrieval method used: {effective_mode}")

    try:
        if effective_mode == "Exact":
            if not exact_request:
                st.warning(
                    "Exact mode needs wording such as: Which columns contain assembly?"
                )
                st.stop()
            field, term = exact_request
            exact_results = exact_search(metadata, field, term)
            st.subheader("Exact Results")
            st.dataframe(exact_results, use_container_width=True, hide_index=True)
            st.caption(f"Exact matches: {len(exact_results)}")
            st.stop()

        if not selected_model:
            st.warning("No Ollama generation model is available.")
            st.stop()

        with st.spinner("Searching metadata..."):
            if effective_mode == "Semantic":
                results = semantic_search(index, metadata, question, candidates)
            else:
                results = hybrid_search(index, metadata, question, candidates)

        if not results:
            st.warning("No metadata records were retrieved.")
            st.stop()

        render_table_overview(results)
        render_table_details(results)

        if show_context:
            st.subheader("Retrieved Context")
            st.code(build_context(results), language="json")

        with st.spinner(f"Asking {selected_model}..."):
            answer = ask_model(selected_model, question, results)

        st.subheader("Answer")
        st.markdown(answer)

    except httpx.ConnectError:
        st.error(f"Could not connect to Ollama at {OLLAMA_BASE_URL}.")
    except httpx.HTTPStatusError as error:
        st.error(f"Ollama returned HTTP {error.response.status_code}.")
        st.code(error.response.text)
    except Exception as error:
        st.error(f"Application error: {error}")
