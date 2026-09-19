# Unity Catalog Metadata Assistant

A local Streamlit application for searching and exploring Databricks Unity Catalog metadata using exact text search, hybrid retrieval, semantic search, and locally hosted Ollama models.

The application uses:

- **Streamlit** for the browser-based interface
- **Ollama** for local embedding and language models
- **nomic-embed-text** for semantic embeddings
- **FAISS** for vector similarity search
- **Keyword retrieval** for literal term matching
- A selectable Ollama generation model, with **qwen3.8:latest** preferred by default when installed

## Architecture

The application is split by responsibility:

```text
uc_app/
├── app.py
├── config.py
├── build_index.py
├── search/
│   ├── __init__.py
│   ├── exact.py
│   ├── keyword.py
│   ├── semantic.py
│   └── hybrid.py
├── llm/
│   ├── __init__.py
│   ├── context.py
│   └── ollama.py
├── ui/
│   ├── __init__.py
│   └── render.py
├── data/
│   ├── uc_prod_2026-09-15_1147_UTC.jsonl
│   ├── uc_metadata.index
│   └── uc_metadata.json
└── README.md
```

### Module responsibilities

| Module | Responsibility |
|---|---|
| `app.py` | Streamlit page flow, search-mode orchestration, input handling, and error handling |
| `config.py` | Shared paths, Ollama URL, model names, timeouts, and defaults |
| `build_index.py` | Generates table embeddings and builds the FAISS index |
| `search/exact.py` | Case-insensitive substring search across all metadata fields |
| `search/keyword.py` | Tokenization, stopwords, weighted keyword matching, and match evidence |
| `search/semantic.py` | Ollama embedding generation and FAISS similarity search |
| `search/hybrid.py` | Combines keyword and semantic rankings using Reciprocal Rank Fusion |
| `llm/context.py` | Builds the metadata context passed to the selected Ollama model |
| `llm/ollama.py` | Retrieves installed models and sends prompts to Ollama |
| `ui/render.py` | Renders result summaries, detailed match evidence, descriptions, and all columns |

## How It Works

### One-time indexing

`build_index.py` reads the table-level JSONL metadata export and creates one embedding per table.

The embedding input includes available metadata such as:

- Catalog name and description
- Schema name and description
- Table name and description
- Column names and descriptions

The indexing process creates:

```text
data/uc_metadata.index
data/uc_metadata.json
```

- `uc_metadata.index` contains the FAISS vectors.
- `uc_metadata.json` contains the metadata records aligned with those vectors.

Rebuild the index whenever the source JSONL metadata changes.

## Search Modes

The application provides three explicit search modes. There is no automatic router.

### Hybrid

Hybrid is the recommended default for natural-language discovery questions.

```text
Question
   ├─ Keyword search
   └─ Semantic search with embeddings and FAISS
              ↓
     Reciprocal Rank Fusion
              ↓
       Candidate tables
              ↓
     Selected Ollama model
              ↓
             Answer
```

Hybrid search:

- Extracts useful terms from the question
- Searches names and descriptions using weighted keyword matching
- Searches semantically similar table embeddings through FAISS
- Merges both rankings
- Sends the selected candidate tables to the chosen Ollama model

Example:

```text
Where are purchase orders stored?
```

### Exact

Exact mode treats the entered value as literal search text, not as a natural-language question.

It performs a case-insensitive substring search across all metadata:

- Catalog names
- Catalog descriptions
- Schema names
- Schema descriptions
- Table names
- Table descriptions
- Column names
- Column descriptions

Example:

```text
cell
```

This can match values such as:

```text
cell
cell2
cell3
battery_cell
cell_temperature
```

Exact mode:

- Searches the complete metadata corpus
- Is not limited by the **Tables considered** setting
- Does not use FAISS
- Does not call the generation model
- Shows every matching table and the metadata locations where the text was found

### Semantic

Semantic mode searches by meaning only.

```text
Question
   ↓
nomic-embed-text
   ↓
Question embedding
   ↓
FAISS similarity search
   ↓
Candidate tables
   ↓
Selected Ollama model
   ↓
Answer
```

Semantic search is useful for fuzzy discovery when the question and metadata may use different terminology.

Example:

```text
Which datasets are related to battery genealogy?
```

Semantic search is ranked rather than exhaustive. A relevant table can be missed if it falls outside the selected candidate count.

## Results Display

### Retrieved Tables

The overview shows:

- Retrieval source
- Similarity score, when applicable
- Keyword score, when applicable
- Catalog
- Schema
- Table
- Table description
- Number of columns

Similarity is a retrieval signal, not proof that a table answers the question.

### Retrieved Table Details

Each result has an expandable details section containing:

- Full table description
- Exact match locations, when available
- Total column count
- Every column and its description

All columns are displayed. The application does not restrict the details view to the first 10 columns.

### Retrieved Context

For Hybrid and Semantic modes, the optional **Show Retrieved Context** setting displays the JSON context sent to the language model.

This is useful for distinguishing:

- Retrieval problems, where irrelevant tables were selected
- Reasoning problems, where the model misinterpreted relevant metadata
- Metadata-quality problems, where names or descriptions are missing

## Tables Considered

The **Tables considered** setting controls how many candidate tables are retained in Hybrid and Semantic modes before answer generation.

- Higher values can improve coverage.
- Higher values also send more metadata to the model and can make responses slower.
- Exact mode searches all metadata, so this setting is disabled for Exact mode.

## Prerequisites

### Ollama

Ollama must be running locally and reachable at the URL configured in `config.py`.

Check the installed models:

```bash
ollama list
```

### Embedding model

Install the embedding model:

```bash
ollama pull nomic-embed-text
```

### Generation model

Install at least one generation model, for example:

```bash
ollama pull qwen3.8
```

The application retrieves installed generation models dynamically from Ollama. Models whose names contain `embed` are excluded from the generation-model dropdown.

## Shared Configuration

Common values should be defined once in `config.py`, for example:

```python
from pathlib import Path

INDEX_FILE = Path("data/uc_metadata.index")
METADATA_FILE = Path("data/uc_metadata.json")

OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text:latest"
PREFERRED_MODEL = "qwen3.8:latest"

REQUEST_TIMEOUT = 600
DEFAULT_CANDIDATES = 25
```

Other modules import these constants instead of defining duplicate values.

## Build the Index

Run:

```bash
uv run build_index.py
```

The script creates or updates:

```text
data/uc_metadata.index
data/uc_metadata.json
```

Never combine an index from one build with metadata from another build.

## Run the Application

Run Streamlit in an isolated UV environment:

```bash
uvx \
  --with streamlit \
  --with faiss-cpu \
  --with httpx \
  --with numpy \
  streamlit run app.py
```

Streamlit normally exposes the application locally at `http://localhost:8501`.

The packages are provided through an isolated UV-managed environment and are not installed globally.

## Example Searches

### Hybrid

```text
Where are purchase orders stored?
```

```text
Which datasets contain supplier capacity information?
```

```text
Which data could support battery traceability analysis?
```

### Exact

```text
assembly
```

```text
purchase_order
```

```text
cell
```

### Semantic

```text
Which datasets are related to production planning?
```

```text
Where could manufacturing genealogy information be stored?
```

## Important Limitations

### Table-level embeddings

The FAISS index contains one embedding per table, not one embedding per column.

Column names and column descriptions are included in each table's embedding text, so columns influence retrieval. FAISS still returns tables as the retrieval unit.

### Semantic and Hybrid retrieval are ranked

Semantic and Hybrid modes return a selected candidate set rather than every possible result. Use Exact mode when exhaustive literal matching is required.

### Metadata quality affects retrieval

The system can only retrieve meaning present in names and descriptions. Missing table descriptions and column descriptions reduce retrieval quality.

A larger or more capable LLM cannot fully compensate for undocumented metadata.

### Source-corpus boundaries

The application can only search metadata present in `uc_metadata.json` and its matching FAISS index.

If temporary or internal tables such as `__materialization_*` were excluded from the original export, the application cannot return them.

### Metadata only

The application searches metadata. It does not query table rows or inspect business data values.

## Troubleshooting

### `Failed to spawn: streamlit`

Use the complete isolated command:

```bash
uvx \
  --with streamlit \
  --with faiss-cpu \
  --with httpx \
  --with numpy \
  streamlit run app.py
```

### `No module named streamlit`

Do not run the Streamlit file as a normal Python script. Start it with the `streamlit run app.py` command shown above.

### `missing ScriptRunContext`

This warning appears when running:

```bash
uv run app.py
```

Use the Streamlit launcher instead.

### Cannot connect to Ollama

Check that Ollama is running:

```bash
ollama list
```

Also verify that `OLLAMA_BASE_URL` in `config.py` is correct.

### Import errors after moving configuration

Modules should import shared constants from `config.py`.

For example, `search/hybrid.py` should import only `semantic_search` from `search.semantic`. Shared values such as `EMBEDDING_MODEL`, `OLLAMA_BASE_URL`, and `REQUEST_TIMEOUT` should come from `config.py`, not from `search.semantic`.

### FAISS index and metadata count do not match

Rebuild both artifacts together:

```bash
uv run build_index.py
```

## Privacy and Security

The application runs locally and sends prompts to the locally configured Ollama service.

Before sharing metadata or generated index files, review them for:

- Email addresses
- Owner identifiers
- Internal catalog and schema names
- Internal table and column descriptions
- Organizational or system-specific information

The metadata export contains no table row values by design, but metadata can still contain sensitive organizational information.

## Potential Next Improvements

- Add `contains`, `whole word`, `starts with`, and `ends with` options to Exact mode
- Add filters for catalog, schema, table type, and documentation status
- Add downloadable CSV output for results
- Add column-level embeddings for more precise field discovery
- Add a metadata-quality dashboard
- Add benchmark questions for comparing local retrieval with other metadata assistants
- Add metadata-change detection and automatic index rebuilding
