import json


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
                    else round(
                        result["similarity"],
                        4,
                    )
                ),
                "keyword_score": result.get(
                    "lexical_score",
                    0,
                ),
                "matched_terms": result.get(
                    "matched_terms",
                    [],
                ),
                "catalog": row.get("catalog"),
                "schema": row.get("schema"),
                "table": row.get("table"),
                "type": row.get("type"),
                "table_comment": row.get(
                    "table_comment"
                ),
                "columns": row.get(
                    "columns",
                    [],
                ),
            }
        )

    return json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
    )