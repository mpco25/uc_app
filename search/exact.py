"""Literal substring search across Unity Catalog metadata."""


def normalize(value):
    """Return a case-insensitive searchable string."""
    if value is None:
        return ""
    return str(value).casefold()


def exact_search(metadata, search_text):
    """
    Search the complete entered text as a case-insensitive substring.

    The search covers:
    - catalog name and description
    - schema name and description
    - table name and description
    - column name and description

    For example, searching for ``cell`` also matches ``cell2``,
    ``battery_cell``, and ``cell_temperature``.

    Every matching table is returned once. ``match_locations`` explains
    exactly where the text occurred within that table's metadata.
    """
    needle = normalize(search_text.strip())

    if not needle:
        return []

    results = []

    for position, record in enumerate(metadata):
        match_locations = []

        record_fields = (
            ("catalog", "Catalog name"),
            ("catalog_comment", "Catalog description"),
            ("schema", "Schema name"),
            ("schema_comment", "Schema description"),
            ("table", "Table name"),
            ("table_comment", "Table description"),
        )

        for field, location in record_fields:
            value = record.get(field)

            if needle in normalize(value):
                match_locations.append(
                    {
                        "field": field,
                        "location": location,
                        "value": value,
                    }
                )

        for column in record.get("columns", []):
            column_name = column.get("name")
            column_comment = column.get("comment")

            if needle in normalize(column_name):
                match_locations.append(
                    {
                        "field": "column_name",
                        "location": "Column name",
                        "column": column_name,
                        "value": column_name,
                    }
                )

            if needle in normalize(column_comment):
                match_locations.append(
                    {
                        "field": "column_comment",
                        "location": "Column description",
                        "column": column_name,
                        "value": column_comment,
                    }
                )

        if not match_locations:
            continue

        results.append(
            {
                "position": position,
                "source": "exact",
                "sources": ["exact"],
                "similarity": None,
                "semantic_rank": None,
                "keyword_rank": None,
                "lexical_score": len(match_locations),
                "hybrid_score": None,
                "matched_terms": [search_text.strip()],
                "match_locations": match_locations,
                "metadata": record,
            }
        )

    return sorted(
        results,
        key=lambda result: (
            result["lexical_score"],
            normalize(result["metadata"].get("table")),
        ),
        reverse=True,
    )
