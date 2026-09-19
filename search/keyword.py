"""Token-based keyword retrieval used by Hybrid search."""

import re


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "do",
    "does",
    "for",
    "from",
    "has",
    "have",
    "how",
    "i",
    "in",
    "information",
    "is",
    "it",
    "me",
    "of",
    "on",
    "or",
    "the",
    "there",
    "to",
    "what",
    "where",
    "which",
    "with",
}


def normalize(value):
    """Return a case-insensitive searchable string."""
    if value is None:
        return ""
    return str(value).casefold()


def tokenize_query(question):
    """Extract meaningful keyword-search terms from a natural-language query."""
    tokens = re.findall(r"[a-z0-9_]+", question.casefold())

    return [
        token
        for token in tokens
        if len(token) > 1 and token not in STOPWORDS
    ]


def keyword_search(metadata, question):
    """
    Search individual query terms across all Unity Catalog metadata.

    Names are weighted more strongly than descriptions. This function is
    intended for Hybrid retrieval. Standalone Exact mode should use the
    complete phrase through ``exact_search`` instead.
    """
    terms = tokenize_query(question)

    if not terms:
        return []

    results = []

    for position, record in enumerate(metadata):
        lexical_score = 0
        matched_terms = set()
        match_locations = []

        record_fields = (
            ("catalog", "Catalog name", 3),
            ("catalog_comment", "Catalog description", 1),
            ("schema", "Schema name", 3),
            ("schema_comment", "Schema description", 1),
            ("table", "Table name", 8),
            ("table_comment", "Table description", 2),
        )

        for term in terms:
            for field, location, weight in record_fields:
                value = record.get(field)
                normalized_value = normalize(value)
                occurrences = normalized_value.count(term)

                if not occurrences:
                    continue

                lexical_score += weight * occurrences
                matched_terms.add(term)
                match_locations.append(
                    {
                        "field": field,
                        "location": location,
                        "term": term,
                        "value": value,
                    }
                )

            for column in record.get("columns", []):
                column_name = column.get("name")
                column_comment = column.get("comment")

                name_occurrences = normalize(column_name).count(term)
                if name_occurrences:
                    lexical_score += 5 * name_occurrences
                    matched_terms.add(term)
                    match_locations.append(
                        {
                            "field": "column_name",
                            "location": "Column name",
                            "term": term,
                            "column": column_name,
                            "value": column_name,
                        }
                    )

                comment_occurrences = normalize(column_comment).count(term)
                if comment_occurrences:
                    lexical_score += 2 * comment_occurrences
                    matched_terms.add(term)
                    match_locations.append(
                        {
                            "field": "column_comment",
                            "location": "Column description",
                            "term": term,
                            "column": column_name,
                            "value": column_comment,
                        }
                    )

        if not lexical_score:
            continue

        results.append(
            {
                "position": position,
                "source": "keyword",
                "sources": ["keyword"],
                "similarity": None,
                "semantic_rank": None,
                "keyword_rank": None,
                "lexical_score": lexical_score,
                "hybrid_score": None,
                "matched_terms": sorted(matched_terms),
                "match_locations": match_locations,
                "metadata": record,
            }
        )

    results.sort(
        key=lambda result: (
            result["lexical_score"],
            len(result["matched_terms"]),
        ),
        reverse=True,
    )

    for rank, result in enumerate(results, start=1):
        result["keyword_rank"] = rank

    return results
