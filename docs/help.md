# Unity Catalog Metadata Assistant

## LLM Model

Select the Ollama model used to generate answers.

Recommended:

- qwen3.8:latest

---

## Search Mode

### Hybrid

Recommended.

Combines:

- keyword matching
- semantic search

and merges the results.

Best for questions such as:

- Where are purchase orders stored?
- Which datasets contain supplier information?

---

### Exact

Literal metadata search.

Searches:

- catalog names
- schema names
- table names
- column names
- descriptions

Example:

cell

will match:

- cell
- cell2
- battery_cell

Exact mode does not use FAISS or the LLM.

---

### Semantic

Meaning-based search using embeddings.

Useful when the exact wording is unknown.

Example:

Which datasets are related to genealogy?

---

## Tables Considered

Controls how many candidate tables are sent to the model.

Higher values:

- may improve coverage
- increase response time

Exact search ignores this setting.

---

## Show Retrieved Context

Displays the JSON sent to the model.

Useful for debugging retrieval quality.

---

## Retrieved Tables

Shows:

- source
- similarity score
- keyword score
- description
- column count

---

## Retrieved Table Details

Shows:

- full table description
- matched metadata locations
- all columns
- column descriptions