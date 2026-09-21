## LLM Model

The selected LLM is used to analyze the retrieved metadata and generate answers.

### Recommended

✅ qwen3.8:latest

Recommended for most users.

- Good reasoning
- Good metadata understanding
- Good balance between quality and speed

### Alternative Models

**devstral-2:latest**
- Strongest reasoning
- Best for difficult or ambiguous questions
- Slowest model

**gemma4:26b**
- Good general-purpose alternative
- Usually concise answers

**phi4:latest**
- Fast responses
- Good for quick exploration

### Other Available Models

Additional models are available and may be useful for experimentation, but are not generally recommended over qwen3.8 for metadata discovery.

### Which model should I use?

For most users:

✅ qwen3.8:latest

For the best possible reasoning:

✅ devstral-2:latest

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

Exact mode does not use AI.

---

### Semantic

Meaning-based search using embeddings.

Useful when the exact wording is unknown.

Example:

Which datasets are related to genealogy?

---

## Tables Considered

When you ask a question, the assistant does not send the entire metadata catalog to the language model.

Instead, the search engine first identifies the most relevant tables. These are called **candidate tables**.

Example:

Question:

Where are purchase orders stored?

The search engine might identify:

- purchasing.purchase_orders
- purchasing.purchase_order_items
- sourcing.suppliers
- logistics.shipments

These candidate tables are then sent to the language model so it can generate an answer.

Higher values:

- may improve coverage by including more potentially relevant tables
- reduce the chance of missing useful metadata
- increase response time because more metadata must be processed

Lower values:

- produce faster responses
- may miss relevant tables that ranked lower in the search results

### Search Mode Behavior

**Hybrid**
- Candidate tables are selected using keyword search and semantic search.

**Semantic**
- Candidate tables are selected using semantic similarity only.

**Exact**
- This setting is ignored because Exact mode searches the entire metadata catalog directly rather than selecting candidate tables first.

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