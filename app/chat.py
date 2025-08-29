from __future__ import annotations

import sys
from typing import Optional

from rich.console import Console
from rich.panel import Panel

from app.retriever import Retriever


def format_answer(hit: dict) -> str:
    answer = str(hit.get("answers", "")).strip()
    tags = str(hit.get("additional_info/tags", "")).strip()
    title = str(hit.get("title/entity_name", "")).strip()
    category = str(hit.get("Category", "")).strip()
    sub_cat = str(hit.get("Sub_Category", "")).strip()
    parts = []
    if title:
        parts.append(f"Title: {title}")
    if category:
        parts.append(f"Category: {category}")
    if sub_cat:
        parts.append(f"Sub-Category: {sub_cat}")
    if answer:
        parts.append(f"Answer: {answer}")
    if tags:
        parts.append(f"Tags: {tags}")
    return "\n".join(parts)


def chat_loop() -> None:
    console = Console()
    retriever = Retriever()
    console.print(Panel("College Placement QA Chatbot - type 'exit' to quit", title="Ready"))
    while True:
        console.print("\n[bold cyan]You:[/bold cyan] ", end="")
        try:
            query = input().strip()
        except EOFError:
            break
        if not query:
            continue
        if query.lower() in {"exit", "quit", "q"}:
            break
        results = retriever.search(query, top_k=3)
        if not results:
            console.print(Panel("I could not find an answer.", title="No Match"))
            continue
        score, hit = results[0]
        console.print(Panel(format_answer(hit), title=f"Match score: {score:.3f}"))


if __name__ == "__main__":
    chat_loop()

