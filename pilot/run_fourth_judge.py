"""
Runs Claude Sonnet 4.6 as a fourth judge on the stratified pilot sample.

Calls the local `claude` CLI in headless mode (subscription auth, OAuth /
keychain). Resumable: skips items already present in the output JSONL.

Usage:
  python pilot/run_fourth_judge.py                         # full run
  python pilot/run_fourth_judge.py --limit 5               # smoke-test
  python pilot/run_fourth_judge.py --workers 4             # parallelism
  python pilot/run_fourth_judge.py --model claude-sonnet-4-6
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
RUBRICS_DIR = ROOT / "rubrics"
JUDGE_SYSTEM_PROMPT = (RUBRICS_DIR / "judge_system_prompt.txt").read_text()
RUBRICS = yaml.safe_load((RUBRICS_DIR / "dimension_rubrics.yaml").read_text())

ITEMS_PATH = ROOT / "pilot" / "fourth_judge_items.jsonl"
SCORES_PATH = ROOT / "pilot" / "fourth_judge_scores.jsonl"
ERRORS_PATH = ROOT / "pilot" / "fourth_judge_errors.jsonl"

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "minimum": 0, "maximum": 3},
        "reasoning": {"type": "string"},
        "evidence": {"type": "string"},
    },
    "required": ["score", "reasoning", "evidence"],
    "additionalProperties": False,
}


def load_few_shot(dimension: str) -> list[dict]:
    path = RUBRICS_DIR / "examples" / f"{dimension}_examples.yaml"
    if not path.exists():
        return []
    return yaml.safe_load(path.read_text()).get("examples", []) or []


def build_rubric_text(dimension: str) -> str:
    dim = RUBRICS[dimension]
    lines = [
        f"DIMENSION: {dim['name']}",
        f"DESCRIPTION: {dim['description']}",
        "",
        "SCORING RUBRIC:",
    ]
    for level in (3, 2, 1, 0):
        entry = dim["rubric"][level]
        lines.append(f"  {level} ({entry['label']}): {entry['description']}")
    return "\n".join(lines)


def build_few_shot_text(examples: list[dict]) -> str:
    lines = ["CALIBRATION EXAMPLES:", ""]
    for i, ex in enumerate(examples, start=1):
        lines.append(f"--- Example {i} ---")
        lines.append(f"User prompt: {ex['user_prompt']}")
        lines.append(f"AI response: {ex['ai_response']}")
        lines.append(f"Correct score: {ex['score']}")
        lines.append(f"Reasoning: {ex['reasoning']}")
        lines.append("")
    return "\n".join(lines)


def build_eval_prompt(item: dict) -> str:
    parts = [build_rubric_text(item["dimension"]), ""]
    fs = load_few_shot(item["dimension"])
    if fs:
        parts.extend([build_few_shot_text(fs), ""])
    parts.append("--- NOW EVALUATE THIS RESPONSE ---")
    parts.append("")
    parts.append(f"USER PROMPT:\n{item['user_prompt']}")
    parts.append("")
    parts.append(f"AI RESPONSE:\n{item['response']}")
    parts.append("")
    parts.append(
        'Score this response. Return ONLY a JSON object with "score" (integer 0-3), '
        '"reasoning" (2-3 sentences), and "evidence" (specific quote or description). '
        "Do not include any text outside the JSON object."
    )
    return "\n".join(parts)


def parse_score_payload(text: str) -> dict | None:
    """Extract the score JSON from Claude's response text."""
    if not text:
        return None
    try:
        obj = json.loads(text)
        if isinstance(obj.get("score"), int) and 0 <= obj["score"] <= 3:
            return obj
    except (json.JSONDecodeError, AttributeError):
        pass
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj.get("score"), int) and 0 <= obj["score"] <= 3:
                return obj
        except json.JSONDecodeError:
            pass
    m = re.search(r"\{[^{}]*\"score\"[^{}]*\}", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj.get("score"), int) and 0 <= obj["score"] <= 3:
                return obj
        except json.JSONDecodeError:
            pass
    m = re.search(r'"?score"?\s*[:=]\s*(\d)', text)
    if m:
        return {
            "score": int(m.group(1)),
            "reasoning": "Parsed from non-JSON response",
            "evidence": text[:200],
        }
    return None


def call_claude(eval_prompt: str, model: str, timeout: int) -> tuple[str | None, str | None]:
    """Invoke `claude -p` once. Returns (assistant_text, error_message)."""
    cmd = [
        "claude",
        "--print",
        "--model",
        model,
        "--no-session-persistence",
        "--output-format",
        "json",
        "--system-prompt",
        JUDGE_SYSTEM_PROMPT,
        "--json-schema",
        json.dumps(JUDGE_SCHEMA),
        eval_prompt,
    ]
    sub_env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=sub_env,
        )
    except subprocess.TimeoutExpired:
        return None, f"timeout after {timeout}s"
    if proc.returncode != 0:
        return None, (
            f"exit={proc.returncode} stderr={proc.stderr[:300]} "
            f"stdout_head={proc.stdout[:300]}"
        )
    out = proc.stdout
    try:
        envelope = json.loads(out)
    except json.JSONDecodeError as e:
        return None, f"envelope_parse_error: {e}; head={out[:200]}"
    if envelope.get("is_error"):
        return None, f"claude error: {envelope.get('result', '')[:300]}"
    structured = envelope.get("structured_output")
    if isinstance(structured, dict) and "score" in structured:
        return json.dumps(structured), None
    result_text = envelope.get("result")
    if not result_text:
        return None, f"no result/structured_output: keys={list(envelope.keys())}"
    return result_text, None


def already_done() -> set[int]:
    if not SCORES_PATH.exists():
        return set()
    done: set[int] = set()
    for line in SCORES_PATH.read_text().splitlines():
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "item_id" in d and d.get("claude_score", -1) >= 0:
            done.add(d["item_id"])
    return done


def load_items() -> list[dict]:
    items = []
    with ITEMS_PATH.open() as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


_write_lock = threading.Lock()


def append_jsonl(path: Path, obj: dict) -> None:
    with _write_lock:
        with path.open("a") as f:
            f.write(json.dumps(obj) + "\n")


def score_item(item: dict, model: str, timeout: int, max_retries: int = 3) -> dict:
    eval_prompt = build_eval_prompt(item)
    last_err = None
    for attempt in range(1, max_retries + 1):
        text, err = call_claude(eval_prompt, model, timeout)
        if err is not None:
            last_err = err
            if attempt < max_retries:
                time.sleep(2 * attempt)
                continue
            return {
                "item_id": item["item_id"],
                "claude_score": -1,
                "error": err,
                "attempts": attempt,
            }
        parsed = parse_score_payload(text)
        if parsed is None:
            last_err = f"parse_failure; head={text[:200]}"
            if attempt < max_retries:
                time.sleep(1)
                continue
            return {
                "item_id": item["item_id"],
                "claude_score": -1,
                "error": last_err,
                "attempts": attempt,
                "raw_result": text,
            }
        return {
            "item_id": item["item_id"],
            "dimension": item["dimension"],
            "model": item["model"],
            "condition": item["condition"],
            "prompt_id": item["prompt_id"],
            "panel_individual_scores": item["panel_individual_scores"],
            "panel_median": item["panel_median"],
            "claude_score": int(parsed["score"]),
            "claude_reasoning": parsed.get("reasoning", ""),
            "claude_evidence": parsed.get("evidence", ""),
            "attempts": attempt,
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--limit", type=int, default=None, help="Score only the first N pending items")
    ap.add_argument("--retries", type=int, default=3)
    args = ap.parse_args()

    items = load_items()
    done = already_done()
    pending = [it for it in items if it["item_id"] not in done]
    if args.limit is not None:
        pending = pending[: args.limit]

    print(f"Total items: {len(items)}")
    print(f"Already scored: {len(done)}")
    print(f"Pending this run: {len(pending)}")
    if not pending:
        print("Nothing to do.")
        return

    t_start = time.time()
    completed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(score_item, it, args.model, args.timeout, args.retries): it
            for it in pending
        }
        for fut in as_completed(futures):
            item = futures[fut]
            try:
                result = fut.result()
            except Exception as e:
                append_jsonl(
                    ERRORS_PATH,
                    {"item_id": item["item_id"], "error": f"unhandled_exception: {e}"},
                )
                completed += 1
                continue
            if result.get("claude_score", -1) < 0:
                append_jsonl(ERRORS_PATH, result)
            else:
                append_jsonl(SCORES_PATH, result)
            completed += 1
            elapsed = time.time() - t_start
            rate = completed / elapsed if elapsed > 0 else 0
            remaining = len(pending) - completed
            eta_min = (remaining / rate / 60) if rate > 0 else float("inf")
            print(
                f"[{completed}/{len(pending)}] item_id={result['item_id']} "
                f"claude={result.get('claude_score')} panel_med={item['panel_median']} "
                f"({rate:.2f}/s, ETA {eta_min:.1f} min)",
                flush=True,
            )

    print(f"\nDone. Wrote scores to {SCORES_PATH}; errors to {ERRORS_PATH}")


if __name__ == "__main__":
    main()
