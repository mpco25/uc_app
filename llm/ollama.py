import httpx

from llm.context import build_context

OLLAMA_BASE_URL = "http://localhost:11434"

def get_models(
    ollama_base_url=OLLAMA_BASE_URL,
):
    response = httpx.get(
        f"{ollama_base_url}/api/tags",
        timeout=30,
    )

    response.raise_for_status()

    return sorted(
        model["name"]
        for model in response.json().get(
            "models",
            [],
        )
        if "embed"
        not in model.get(
            "name",
            "",
        ).lower()
    )


def ask_model(
    model,
    question,
    results,
    ollama_base_url,
    request_timeout,
):
    context = build_context(
        results
    )

    response = httpx.post(
        f"{ollama_base_url}/api/chat",
        json={
            "model": model,
            "stream": False,
            "keep_alive": "10m",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a Databricks "
                        "Unity Catalog metadata "
                        "assistant."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question:\n"
                        f"{question}\n\n"
                        f"Metadata:\n"
                        f"{context}"
                    ),
                },
            ],
            "options": {
                "temperature": 0,
            },
        },
        timeout=request_timeout,
    )

    response.raise_for_status()

    answer = (
        response.json()
        .get("message", {})
        .get("content")
    )

    if not answer:
        raise RuntimeError(
            "Ollama returned no answer."
        )

    return answer.strip()