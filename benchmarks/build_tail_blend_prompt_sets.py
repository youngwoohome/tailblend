# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build JSONL prompt sets for BoN scheduler generalization experiments.

The benchmark_taper_plus.py driver only needs one ``prompt`` field per row.
This helper converts a few public datasets into that common format while
keeping prompts within a configurable token budget.
"""

from __future__ import annotations

import argparse
import json
import os
import zipfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def truncate_by_tokens(tokenizer: Any, text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return text
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= max_tokens:
        return text
    return tokenizer.decode(token_ids[:max_tokens], skip_special_tokens=True)


def gsm8k_rows(limit: int, tokenizer: Any, max_prompt_tokens: int):
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test")
    for idx, row in enumerate(ds):
        prompt = (
            "Solve the following grade-school math word problem. "
            "Show your reasoning step by step and give the final answer.\n\n"
            f"Problem:\n{row['question']}\n\nAnswer:"
        )
        yield {
            "dataset": "openai/gsm8k",
            "id": f"gsm8k-test-{idx}",
            "prompt": truncate_by_tokens(tokenizer, prompt, max_prompt_tokens),
        }
        if idx + 1 >= limit:
            return


def mbpp_rows(limit: int, tokenizer: Any, max_prompt_tokens: int):
    from datasets import load_dataset

    ds = load_dataset("google-research-datasets/mbpp", "full", split="test")
    for out_idx, row in enumerate(ds):
        tests = "\n".join(row.get("test_list") or [])
        prompt = (
            "Write a correct Python function for the following programming "
            "task. Return only code, with no markdown fences.\n\n"
            f"Task:\n{row['text']}\n\n"
            f"Unit tests:\n{tests}\n\nCode:"
        )
        yield {
            "dataset": "google-research-datasets/mbpp",
            "id": f"mbpp-test-{row.get('task_id', out_idx)}",
            "prompt": truncate_by_tokens(tokenizer, prompt, max_prompt_tokens),
        }
        if out_idx + 1 >= limit:
            return


def longbench_rows(
    limit: int,
    tokenizer: Any,
    max_prompt_tokens: int,
    subset: str,
):
    zip_path = hf_hub_download(
        repo_id="THUDM/LongBench",
        repo_type="dataset",
        filename="data.zip",
    )
    member = f"data/{subset}.jsonl"
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(zip_path) as z:
        with z.open(member) as f:
            for line in f:
                row = json.loads(line)
                rows.append(row)

    # Use the longest examples first so this really stresses prefill/context.
    rows.sort(key=lambda row: int(row.get("length") or 0), reverse=True)
    for idx, row in enumerate(rows[:limit]):
        question = row.get("input") or "Summarize the document."
        prompt = (
            "Answer the question using the long context below. Be concise but "
            "include the key evidence.\n\n"
            f"Context:\n{row.get('context', '')}\n\n"
            f"Question:\n{question}\n\nAnswer:"
        )
        yield {
            "dataset": f"THUDM/LongBench/{subset}",
            "id": f"longbench-{subset}-{row.get('_id', idx)}",
            "source_length": row.get("length"),
            "prompt": truncate_by_tokens(tokenizer, prompt, max_prompt_tokens),
        }


def _messages_to_prompt(messages: list[dict[str, Any]], max_turns: int) -> str:
    trimmed = messages[: max(1, max_turns)]
    parts: list[str] = []
    for message in trimmed:
        role = str(message.get("role") or "user").strip().title()
        content = str(message.get("content") or "").strip()
        if content:
            parts.append(f"{role}: {content}")
    parts.append("Assistant:")
    return "\n\n".join(parts)


def lmsys_rows(limit: int, tokenizer: Any, max_prompt_tokens: int):
    from datasets import load_dataset

    if not os.getenv("HF_TOKEN"):
        raise RuntimeError(
            "lmsys/lmsys-chat-1m is gated and needs HF_TOKEN."
        )
    ds = load_dataset("lmsys/lmsys-chat-1m", split="train", streaming=True)
    count = 0
    for idx, row in enumerate(ds):
        messages = row.get("conversation") or row.get("messages") or []
        if not messages:
            continue
        prompt = _messages_to_prompt(messages, max_turns=8)
        yield {
            "dataset": "lmsys/lmsys-chat-1m",
            "id": f"lmsys-{idx}",
            "prompt": truncate_by_tokens(tokenizer, prompt, max_prompt_tokens),
        }
        count += 1
        if count >= limit:
            return


def ultrachat_rows(limit: int, tokenizer: Any, max_prompt_tokens: int):
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    for idx, row in enumerate(ds):
        messages = row.get("messages") or []
        prompt = _messages_to_prompt(messages, max_turns=8)
        yield {
            "dataset": "HuggingFaceH4/ultrachat_200k",
            "id": f"ultrachat-{row.get('prompt_id', idx)}",
            "prompt": truncate_by_tokens(tokenizer, prompt, max_prompt_tokens),
        }
        if idx + 1 >= limit:
            return


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument(
        "--output-dir",
        default="benchmark_outputs/tail_blend_generalization_prompts",
    )
    parser.add_argument("--max-short-prompt-tokens", type=int, default=896)
    parser.add_argument("--max-long-prompt-tokens", type=int, default=4096)
    parser.add_argument("--longbench-subset", default="qasper")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    jobs = [
        (
            "gsm8k",
            output_dir / "gsm8k_40.jsonl",
            gsm8k_rows(args.limit, tokenizer, args.max_short_prompt_tokens),
        ),
        (
            "mbpp",
            output_dir / "mbpp_40.jsonl",
            mbpp_rows(args.limit, tokenizer, args.max_short_prompt_tokens),
        ),
        (
            f"longbench_{args.longbench_subset}",
            output_dir / f"longbench_{args.longbench_subset}_40.jsonl",
            longbench_rows(
                args.limit,
                tokenizer,
                args.max_long_prompt_tokens,
                args.longbench_subset,
            ),
        ),
    ]

    if os.getenv("HF_TOKEN"):
        jobs.append(
            (
                "lmsys_chat_1m",
                output_dir / "lmsys_chat_1m_40.jsonl",
                lmsys_rows(args.limit, tokenizer, args.max_short_prompt_tokens),
            )
        )
    else:
        print("lmsys_chat_1m: skipped (set HF_TOKEN to access gated dataset)")

    for name, path, rows in jobs:
        try:
            count = write_jsonl(path, rows)
            print(f"{name}: wrote {count} prompts to {path}")
        except RuntimeError as exc:
            if name == "lmsys_chat_1m":
                print(f"{name}: skipped ({exc})")
            else:
                raise

    fallback_path = output_dir / "chat_ultrachat_40.jsonl"
    count = write_jsonl(
        fallback_path,
        ultrachat_rows(args.limit, tokenizer, args.max_short_prompt_tokens),
    )
    print(f"chat_ultrachat: wrote {count} prompts to {fallback_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
