"""Streamlit rendering helpers for search results."""

import streamlit as st


def render_table_overview(results):
    """Render a compact table-level overview of retrieved metadata."""
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


def render_table_details(
    results,
):
    """Render expandable evidence and column details for each result."""
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

                st.dataframe(
                    [
                        {
                            "Column": column.get("name"),
                            "Description": (
                                column.get("comment")
                                or "(No description)"
                            ),
                        }
                        for column in columns
                    ],
                    use_container_width=True,
                    hide_index=True,
                )


def render_exact_summary(results, search_text):
    """Render counts and scope for an exhaustive exact-text search."""
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
