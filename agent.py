"""BioQuery Agent: an LLM that decides which NCBI tools to call, then
synthesizes a cited answer from what the tools return.

Supports both Anthropic Claude API (default) and local Ollama.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field

from tools import TOOL_SCHEMAS, dispatch

DEFAULT_MODEL = os.getenv("BIOQUERY_MODEL", "claude-3-5-sonnet-20241022")
MAX_TOKENS = 2048

SYSTEM_PROMPT = """You are a biomedical research assistant. Scientists ask you \
questions in plain English and you answer using public NCBI data.

How to work:
1. Decide which tools are needed. Use get_gene_info to confirm genes, \
search_pubmed for published evidence, and search_geo for expression datasets. \
Call several tools when the question has several parts.
2. Use only facts that appear in tool results. Never invent PMIDs, accessions, \
or findings. If the tools return little or nothing, say so plainly.
3. Combine the sources into one coherent answer. If sources disagree or only \
partly answer the question, point that out.
4. Cite every claim inline with its identifier, e.g. (PMID 12345678), \
(GSE12345), or (Gene ID 3596).
5. Keep the answer concise: a short summary first, then supporting detail, then \
a one-line note on limitations."""


@dataclass
class AgentResult:
    answer: str = ""
    tool_calls: list = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    turns: int = 0


def run_agent(
    question: str,
    client=None,
    model: str = DEFAULT_MODEL,
    max_turns: int = 8,
    verbose: bool = False,
    dispatch_fn=dispatch,
) -> AgentResult:
    """Run the tool-use loop using either Anthropic or Ollama/OpenAI backend."""
    use_anthropic = os.getenv("ANTHROPIC_API_KEY") or client is not None and type(client).__name__ == "Anthropic"
    
    # Auto-detect client if not provided
    if client is None:
        if os.getenv("ANTHROPIC_API_KEY"):
            import anthropic
            client = anthropic.Anthropic()
        else:
            from openai import OpenAI
            client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
            if model == DEFAULT_MODEL and "claude" in DEFAULT_MODEL:
                model = "llama3.2"  # Fallback model for local Ollama if no Anthropic key

    result = AgentResult()
    def log(message: str) -> None:
        if verbose:
            print(message, file=sys.stderr)

    # --- ANTHROPIC CLIENT PATH ---
    if hasattr(client, "messages"):
        messages = [{"role": "user", "content": question}]
        def _text_of(response) -> str:
            return "".join(b.text for b in response.content if b.type == "text").strip()

        for turn in range(1, max_turns + 1):
            response = client.messages.create(
                model=model,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                tools=TOOL_SCHEMAS,
                messages=messages,
            )
            result.turns = turn
            result.input_tokens += response.usage.input_tokens
            result.output_tokens += response.usage.output_tokens
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason != "tool_use":
                result.answer = _text_of(response)
                return result

            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                log(f"[turn {turn}] tool call: {block.name}({block.input})")
                output = dispatch_fn(block.name, block.input)
                result.tool_calls.append({"name": block.name, "input": dict(block.input)})
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": output}
                )
            messages.append({"role": "user", "content": tool_results})

        messages.append({"role": "user", "content": "The tool budget is used up. Write your final answer now."})
        response = client.messages.create(
            model=model, max_tokens=MAX_TOKENS, system=SYSTEM_PROMPT, tools=TOOL_SCHEMAS, tool_choice={"type": "none"}, messages=messages
        )
        result.turns += 1
        result.input_tokens += response.usage.input_tokens
        result.output_tokens += response.usage.output_tokens
        result.answer = _text_of(response)
        return result

    # --- OPENAI / OLLAMA CLIENT PATH ---
    else:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        for turn in range(1, max_turns + 1):
            response = client.chat.completions.create(
                model=model, max_tokens=MAX_TOKENS, tools=TOOL_SCHEMAS, messages=messages
            )
            result.turns = turn
            if hasattr(response, "usage") and response.usage:
                result.input_tokens += getattr(response.usage, "prompt_tokens", 0)
                result.output_tokens += getattr(response.usage, "completion_tokens", 0)

            message = response.choices[0].message
            messages.append(message)

            if not message.tool_calls:
                result.answer = message.content or ""
                return result

            for tool_call in message.tool_calls:
                func_name = tool_call.function.name
                try:
                    func_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    func_args = {}
                log(f"[turn {turn}] tool call: {func_name}({func_args})")
                output = dispatch_fn(func_name, func_args)
                result.tool_calls.append({"name": func_name, "input": func_args})
                messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": output})

        messages.append({"role": "user", "content": "The tool budget is used up. Write your final answer now."})
        response = client.chat.completions.create(model=model, max_tokens=MAX_TOKENS, tools=None, messages=messages)
        result.turns += 1
        if hasattr(response, "usage") and response.usage:
            result.input_tokens += getattr(response.usage, "prompt_tokens", 0)
            result.output_tokens += getattr(response.usage, "completion_tokens", 0)
        result.answer = response.choices[0].message.content or ""
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask a biology question; the agent queries NCBI.")
    parser.add_argument("question", nargs="*", help="Your question in plain English.")
    parser.add_argument("--max-turns", type=int, default=8, help="Max model/tool rounds (default 8).")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model name (default {DEFAULT_MODEL}).")
    parser.add_argument("--quiet", action="store_true", help="Hide tool-call logging.")
    args = parser.parse_args()

    question = " ".join(args.question).strip() or input("Question: ").strip()
    if not question:
        print("No question given.", file=sys.stderr)
        return 1

    result = run_agent(question, model=args.model, max_turns=args.max_turns, verbose=not args.quiet)
    print(result.answer)
    print(
        f"\n--- {len(result.tool_calls)} tool call(s), {result.turns} turn(s), "
        f"{result.input_tokens} input / {result.output_tokens} output tokens ---",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())