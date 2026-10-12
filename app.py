"""Command-line chat entry point for the medical research assistant."""

from __future__ import annotations

import argparse

from agent.chat_service import ask_agent, ensure_knowledge_base


EXIT_COMMANDS = {"exit", "quit", "q", "退出"}


def main() -> None:
    """Run one-shot or interactive chat mode."""

    parser = argparse.ArgumentParser(description="Medical AI research assistant")
    parser.add_argument("--query", "-q", help="Ask one question and exit.")
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Start an interactive chat loop. This is the default when --query is omitted.",
    )
    args = parser.parse_args()

    stats = ensure_knowledge_base()
    if stats:
        print(
            "Knowledge base built: "
            f"{stats['markdown_files']} files, {stats['chunks']} chunks, "
            f"{stats['added']} vectors added."
        )

    if args.query:
        _ask_once(args.query)
        return

    _interactive_loop()


def _ask_once(query: str) -> None:
    response = ask_agent(query)
    print(response.answer)


def _interactive_loop() -> None:
    print("Medical Research Agent chat mode")
    print("Type your question. Use 'exit' or '退出' to stop.\n")

    while True:
        try:
            query = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nStopped.")
            return

        if not query:
            continue
        if query.lower() in EXIT_COMMANDS:
            print("Stopped.")
            return

        try:
            response = ask_agent(query)
        except Exception as exc:
            print(f"Agent error: {exc}")
            continue

        print("\nAgent>\n")
        print(response.answer)
        print("")


if __name__ == "__main__":
    main()

