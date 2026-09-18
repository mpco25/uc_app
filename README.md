# Unity Catalog Metadata Assistant

A local Streamlit application for asking natural-language questions about Databricks Unity Catalog metadata.

The application uses:

- **Streamlit** for the browser-based user interface
- **Ollama** for local embedding and language models
- **nomic-embed-text** to convert questions and metadata into embeddings
- **FAISS** to retrieve semantically similar Unity Catalog tables
- A selectable Ollama model, such as **qwen3.8**, to generate the final answer

## How It Works

### One-time indexing

The separate `build_index.py` script reads the Unity Catalog JSONL export and generates one embedding per table.

Each embedded table record includes available metadata such as:

- Catalog name
- Schema name
- Table name
- Table description or comment
- Column names
- Column descriptions or comments

The indexing process creates:

```text
uc_metadata.index
uc_metadata.json
```

`uc_metadata.index` contains the FAISS vectors.

`uc_metadata.json` contains the metadata records corresponding to those vectors.

### Question answering

When a user submits a question, the application follows this flow:

```text
Question
   ↓
nomic-embed-text
   ↓
Question embedding
   ↓
FAISS similarity search
   ↓
Most relevant table metadata
   ↓
Selected Ollama language model
   ↓
Answer
```

Only the retrieved metadata is sent to the selected language model. The full metadata export is not sent with every question.

## Project Files

Keep these files in the same directory:

```text
app.py
build_index.py
uc_prod_2026-09-15_1147_UTC.jsonl
uc_metadata.index
uc_metadata.json
README.md
```

### File purposes

| File | Purpose |
|---|---|
| `app.py` | Streamlit user interface and question-answering workflow |
| `build_index.py` | Builds embeddings and the FAISS index |
| `uc_prod_2026-09-15_1147_UTC.jsonl` | Original table-level Unity Catalog metadata export |
| `uc_metadata.index` | FAISS vector index |
| `uc_metadata.json` | Metadata records aligned with the FAISS vectors |
| `README.md` | Project documentation |

## Prerequisites

### 1. Ollama

Ollama must be running locally and reachable at:

```text
http://localhost:11434
```

Check that it is available:

```bash
ollama list
```

### 2. Embedding model

The application expects:

```text
nomic-embed-text:latest
```

Install it if necessary:

```bash
ollama pull nomic-embed-text
```

### 3. Answering model

Install at least one generation model, for example:

```bash
ollama pull qwen3.8
```

The UI retrieves installed models dynamically from Ollama's `/api/tags` endpoint. Models whose names contain `embed` are excluded from the answering-model dropdown.

## Build the Index

Run the indexing script once before starting the application:

```bash
uv run build_index.py
```

This creates:

```text
uc_metadata.index
uc_metadata.json
```

Rebuild the index whenever the source JSONL metadata changes.

## Run the Streamlit Application

The application can run in an isolated environment without globally installing Python packages:

```bash
uvx \
  --with streamlit \
  --with faiss-cpu \
  --with httpx \
  --with numpy \
  streamlit run app.py
```

Streamlit will display a local browser address, normally:

```text
http://localhost:8501
```

Nothing is installed into the global Python environment. UV stores downloaded packages in its cache and runs them in an isolated environment.

## Using the Application

1. Start Ollama.
2. Start the Streamlit application.
3. Select an installed Ollama model from the sidebar.
4. Enter a metadata question.
5. Select a search mode.
6. Click **Ask**.
7. Review the retrieved tables and the generated answer.

Example questions:

```text
Where are purchase orders stored?
```

```text
Which tables contain information about supplier capacity?
```

```text
Is there an identifier for an assembly line in any table?
```

```text
Which datasets are related to battery traceability?
```

## Understanding the Results

### Retrieved tables

The application displays the tables returned by FAISS together with their similarity scores.

A higher similarity score indicates that the table metadata is more semantically similar to the question. A similarity score is a retrieval signal, not proof that the table answers the question.

### Generated answer

The selected Ollama model receives the user question and the retrieved metadata. It is instructed to avoid inventing catalogs, schemas, tables, columns, or relationships.

Always inspect the retrieved tables when an answer appears incorrect. This helps distinguish between:

- A retrieval problem, where FAISS returned irrelevant tables
- A reasoning problem, where the model misinterpreted relevant metadata
- A source-data problem, where descriptions or comments are missing

## Search Modes and Current Status

The UI contains the following search-mode choices:

- Auto
- Semantic
- Exact
- Hybrid

In the current `app.py` version, the retrieval implementation is **semantic search**. The other modes are visible in the UI but are not yet implemented as distinct retrieval strategies.

This means:

- Selecting Auto currently uses semantic retrieval
- Selecting Exact currently uses semantic retrieval
- Selecting Hybrid currently uses semantic retrieval
- Selecting Semantic uses semantic retrieval

Do not treat Exact or Hybrid as functional until their retrieval logic is added.

## Important Limitations

### Retrieval is table-level

The index contains one vector per table, not one vector per column.

Column names and comments are included in the table text used to create the embedding, so they influence retrieval. However, FAISS returns tables rather than individual columns.

### Semantic search is not exhaustive

FAISS returns the top matching tables. It does not guarantee that every exact match is returned.

For example, a request such as:

```text
List every column whose name contains assembly.
```

is better handled by an exhaustive exact scan of all column names than by top-N semantic retrieval.

### Source corpus differences

The application can only search metadata present in `uc_metadata.json` and its matching FAISS index.

If internal or temporary tables such as `__materialization_*` were excluded from the original export, the application cannot return them. Results should therefore not be compared directly with another tool unless both tools search the same metadata corpus.

### Metadata only

The application searches metadata. It does not query table rows or inspect business data values.

## Recommended Next Improvements

1. Implement exact search for deterministic requests such as `contains`, `starts with`, and `ends with`.
2. Implement hybrid retrieval by combining exact keyword matches with FAISS semantic matches.
3. Add automatic query routing between exact, semantic, and hybrid retrieval.
4. Add column-level embeddings if table-level retrieval misses important fields.
5. Add downloadable CSV output for retrieved table and column results.
6. Add search history and saved benchmark questions.
7. Add metadata refresh detection and automatic index rebuilding.

## Troubleshooting

### `Failed to spawn: streamlit`

Run Streamlit through `uvx` with all required packages:

```bash
uvx \
  --with streamlit \
  --with faiss-cpu \
  --with httpx \
  --with numpy \
  streamlit run app.py
```

### `No module named streamlit`

Do not run the file as a normal Python script. Use the full `uvx ... streamlit run app.py` command shown above.

### `missing ScriptRunContext`

This occurs when running:

```bash
uv run app.py
```

A Streamlit application must be launched with:

```bash
streamlit run app.py
```

Use the isolated `uvx` command documented above.

### Cannot connect to Ollama

Check that Ollama is running:

```bash
ollama list
```

The application expects Ollama at:

```text
http://localhost:11434
```

### FAISS index and metadata count do not match

Rebuild both files together:

```bash
uv run build_index.py
```

Never combine `uc_metadata.index` from one build with `uc_metadata.json` from another build.

## Privacy and Security

The application runs locally and sends prompts to the locally configured Ollama service.

Before sharing the source metadata or generated index files, review them for:

- Email addresses
- Owner identifiers
- Internal catalog and schema names
- Internal table and column descriptions
- Organizational or system-specific details

The metadata export contains no table row values by design, but metadata itself can still contain sensitive organizational information.