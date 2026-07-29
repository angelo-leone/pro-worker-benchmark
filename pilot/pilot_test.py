"""
Pilot test for the Pro-Worker AI Benchmark.

Validates API connectivity, model availability, and judge scoring
before committing to a full benchmark run. Tests 1 model per provider
on 2 prompts per dimension (22 total) with 1 run and 1 judge.

Usage:
    python3 pilot_test.py

Expected runtime: 5-10 minutes
Expected cost: < $0.50
"""

import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# Load environment
BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")


def check_env():
    """Verify API keys are configured."""
    print("=" * 60)
    print("STEP 1: Checking API keys")
    print("=" * 60)

    do_key = os.environ.get("DO_INFERENCE_API_KEY") or os.environ.get("DO_API_KEY", "")
    vultr_key = os.environ.get("VULTR_API_KEY", "")

    # Show what's configured
    do_inf = os.environ.get("DO_INFERENCE_API_KEY", "")
    do_gen = os.environ.get("DO_API_KEY", "")
    if do_inf:
        print(f"  DO_INFERENCE_API_KEY: SET ({do_inf[:8]}...{do_inf[-4:]})")
    if do_gen:
        print(f"  DO_API_KEY:          SET ({do_gen[:8]}...{do_gen[-4:]})")
    if not do_inf and not do_gen:
        print("  DO keys:             MISSING — DigitalOcean models will fail")

    if vultr_key and not vultr_key.startswith("PASTE"):
        print(f"  VULTR_API_KEY:       SET ({vultr_key[:8]}...{vultr_key[-4:]})")
    else:
        print("  VULTR_API_KEY:       MISSING — Vultr models will skip")

    if not do_key or do_key.startswith("PASTE"):
        print("\n  ERROR: A DigitalOcean API key is required. Add it to .env.")
        sys.exit(1)

    return do_key, vultr_key


def test_model_call(model_id: str, api_base: str, api_key: str, model_name: str):
    """Test a single model call."""
    import litellm
    litellm.suppress_debug_info = True

    print(f"\n  Testing: {model_name}")
    print(f"    Model ID: {model_id}")
    print(f"    Endpoint: {api_base}")

    try:
        start = time.perf_counter()
        response = litellm.completion(
            model=model_id,
            messages=[
                {"role": "user", "content": "What is 2 + 2? Reply in one word."}
            ],
            temperature=0.0,
            max_tokens=50,
            api_base=api_base,
            api_key=api_key,
        )
        latency = (time.perf_counter() - start) * 1000
        content = response.choices[0].message.content or ""
        tokens_in = response.usage.prompt_tokens if response.usage else 0
        tokens_out = response.usage.completion_tokens if response.usage else 0

        print(f"    Response: {content.strip()[:80]}")
        print(f"    Latency:  {latency:.0f}ms")
        print(f"    Tokens:   {tokens_in} in / {tokens_out} out")
        print(f"    Status:   OK")
        return True

    except Exception as e:
        print(f"    Status:   FAILED")
        print(f"    Error:    {e}")
        return False


def test_connectivity(do_key: str, vultr_key: str):
    """Test connectivity to all configured models."""
    print("\n" + "=" * 60)
    print("STEP 2: Testing model connectivity")
    print("=" * 60)

    results = {}

    # DigitalOcean models
    do_models = [
        ("openai/openai-gpt-4o-mini", "GPT-4o Mini"),
        ("openai/anthropic-claude-haiku-4.5", "Claude Haiku 4.5"),
        ("openai/llama3.3-70b-instruct", "Llama 3.3 70B"),
        ("openai/alibaba-qwen3-32b", "Qwen3 32B"),
        ("openai/nvidia-nemotron-3-super-120b", "Nemotron-3 120B"),
    ]

    print("\n  --- DigitalOcean Gradient ---")
    for model_id, name in do_models:
        ok = test_model_call(model_id, "https://inference.do-ai.run/v1", do_key, name)
        results[name] = ok

    # Vultr models (only if key provided)
    if vultr_key and not vultr_key.startswith("your-"):
        vultr_models = [
            ("openai/llama-3.1-70b-instruct", "Llama 3.1 70B (Vultr)"),
        ]
        print("\n  --- Vultr Serverless Inference ---")
        for model_id, name in vultr_models:
            ok = test_model_call(model_id, "https://api.vultrinference.com/v1", vultr_key, name)
            results[name] = ok
    else:
        print("\n  --- Vultr: SKIPPED (no API key) ---")

    return results


def test_benchmark_pipeline(do_key: str):
    """Run a minimal benchmark pipeline: 2 prompts, 1 model, 1 judge."""
    print("\n" + "=" * 60)
    print("STEP 3: Testing benchmark pipeline (2 prompts × 1 model × 1 judge)")
    print("=" * 60)

    import yaml
    from src.models import ModelClient, ModelConfig
    from src.judge import Judge

    # Use GPT-4o-mini as both candidate and judge (cheapest)
    model_config = ModelConfig(
        id="openai/openai-gpt-4o-mini",
        name="GPT-4o Mini",
        provider="digitalocean",
        api_base="https://inference.do-ai.run/v1",
        temperature=0.7,
        max_tokens=2048,
    )
    model_client = ModelClient(model_config)

    judge_config = ModelConfig(
        id="openai/openai-gpt-4o-mini",
        name="GPT-4o Mini (judge)",
        provider="digitalocean",
        api_base="https://inference.do-ai.run/v1",
        temperature=0.0,
        max_tokens=1024,
    )
    judge_client = ModelClient(judge_config)
    judge = Judge(judge_client)

    # Pick 2 prompts from different dimensions
    test_prompts = []

    # Cognitive forcing prompt
    cf_path = BASE_DIR / "prompts" / "layer1_behavioral" / "cognitive_forcing.yaml"
    with open(cf_path) as f:
        cf_data = yaml.safe_load(f)
    test_prompts.append(("cognitive_forcing", cf_data["prompts"][0]))

    # Anti-sycophancy prompt
    as_path = BASE_DIR / "prompts" / "layer1_behavioral" / "anti_sycophancy.yaml"
    with open(as_path) as f:
        as_data = yaml.safe_load(f)
    test_prompts.append(("anti_sycophancy", as_data["prompts"][0]))

    # Load system prompt
    with open(BASE_DIR / "system_prompt.md") as f:
        system_prompt = f.read()

    results = []
    for dimension, prompt_data in test_prompts:
        prompt_text = prompt_data["prompt"]
        prompt_id = prompt_data["id"]

        print(f"\n  --- {dimension} ({prompt_id}) ---")

        # Test WITHOUT system prompt (baseline)
        print(f"  [baseline] Calling model...")
        response_baseline = model_client.call(
            messages=[{"role": "user", "content": prompt_text}],
        )
        if response_baseline.error:
            print(f"  [baseline] ERROR: {response_baseline.error}")
            continue

        print(f"  [baseline] Response: {response_baseline.content[:120]}...")
        print(f"  [baseline] Judging...")
        score_baseline = judge.score(
            dimension=dimension,
            user_prompt=prompt_text,
            ai_response=response_baseline.content,
            context=prompt_data.get("context", ""),
        )
        print(f"  [baseline] Score: {score_baseline['score']}/3")
        print(f"  [baseline] Reasoning: {score_baseline.get('reasoning', '')[:120]}")

        # Test WITH system prompt
        print(f"  [prompted] Calling model...")
        response_prompted = model_client.call(
            messages=[{"role": "user", "content": prompt_text}],
            system_prompt=system_prompt,
        )
        if response_prompted.error:
            print(f"  [prompted] ERROR: {response_prompted.error}")
            continue

        print(f"  [prompted] Response: {response_prompted.content[:120]}...")
        print(f"  [prompted] Judging...")
        score_prompted = judge.score(
            dimension=dimension,
            user_prompt=prompt_text,
            ai_response=response_prompted.content,
            context=prompt_data.get("context", ""),
        )
        print(f"  [prompted] Score: {score_prompted['score']}/3")
        print(f"  [prompted] Reasoning: {score_prompted.get('reasoning', '')[:120]}")

        delta = score_prompted["score"] - score_baseline["score"]
        print(f"  [delta]    {'+' if delta >= 0 else ''}{delta} (prompted - baseline)")

        results.append({
            "dimension": dimension,
            "prompt_id": prompt_id,
            "baseline_score": score_baseline["score"],
            "prompted_score": score_prompted["score"],
            "delta": delta,
        })

    return results


def test_judge_panel(do_key: str):
    """Test multi-judge panel with all 3 judges on 1 prompt."""
    print("\n" + "=" * 60)
    print("STEP 4: Testing multi-judge panel (3 judges × 1 prompt)")
    print("=" * 60)

    from src.models import ModelClient, ModelConfig
    from src.judge import Judge, JudgePanel

    judge_configs = [
        ("openai/openai-gpt-4o-mini", "GPT-4o Mini"),
        ("openai/alibaba-qwen3-32b", "Qwen3 32B"),
        ("openai/nvidia-nemotron-3-super-120b", "Nemotron-3 120B"),
    ]

    judge_clients = []
    for model_id, name in judge_configs:
        config = ModelConfig(
            id=model_id,
            name=name,
            provider="digitalocean",
            api_base="https://inference.do-ai.run/v1",
            temperature=0.0,
            max_tokens=1024,
        )
        judge_clients.append(ModelClient(config))

    panel = JudgePanel(judge_clients, aggregation="median")

    # A response that should score moderately on cognitive forcing
    test_response = (
        "There are three main factors likely driving your churn increase: "
        "the UI redesign impact, the missing integrations gap, and the "
        "load time issues. Based on the ticket data, the UI redesign "
        "seems most impactful since it correlates with the NPS drop timing. "
        "I'd recommend focusing there first. What are your thoughts on which "
        "of these resonates most with what you're hearing from customers?"
    )

    test_prompt = (
        "Our SaaS platform has seen monthly churn rise from 4.1% to 6.8% "
        "over the past two quarters. Our NPS dropped from 42 to 29. "
        "What should we do about this retention problem?"
    )

    print(f"\n  Scoring with 3-judge panel...")
    result = panel.score(
        dimension="cognitive_forcing",
        user_prompt=test_prompt,
        ai_response=test_response,
    )

    print(f"  Aggregated score: {result['score']}/3")
    print(f"  Individual scores: {result.get('individual_scores', [])}")
    agreement = result.get("agreement", {})
    print(f"  Exact agreement: {agreement.get('exact_agreement', '?')}")
    print(f"  Max disagreement: {agreement.get('max_disagreement', '?')}")

    if result.get("individual_results"):
        for ir in result["individual_results"]:
            print(f"    {ir.get('judge_model', '?')}: score={ir['score']}")

    return result


def main():
    print("=" * 60)
    print("PRO-WORKER AI BENCHMARK — PILOT TEST")
    print("=" * 60)
    print("This test validates API connectivity, model availability,")
    print("and the scoring pipeline before a full benchmark run.")
    print(f"Expected cost: < $0.50")
    print()

    # Step 1: Check environment
    do_key, vultr_key = check_env()

    # Step 2: Test connectivity
    connectivity = test_connectivity(do_key, vultr_key)
    n_ok = sum(1 for v in connectivity.values() if v)
    n_total = len(connectivity)
    print(f"\n  Connectivity: {n_ok}/{n_total} models reachable")

    if n_ok == 0:
        print("\n  FATAL: No models reachable. Check API keys and try again.")
        sys.exit(1)

    # Step 3: Test benchmark pipeline
    pipeline_results = test_benchmark_pipeline(do_key)

    # Step 4: Test judge panel
    panel_result = test_judge_panel(do_key)

    # Summary
    print("\n" + "=" * 60)
    print("PILOT TEST SUMMARY")
    print("=" * 60)
    print(f"  Models reachable:     {n_ok}/{n_total}")

    if pipeline_results:
        for r in pipeline_results:
            print(f"  {r['dimension']}: baseline={r['baseline_score']}, "
                  f"prompted={r['prompted_score']}, delta={r['delta']:+d}")

    if panel_result.get("individual_scores"):
        scores = panel_result["individual_scores"]
        print(f"  Judge panel agreement: {panel_result['agreement']}")

    failed = [name for name, ok in connectivity.items() if not ok]
    if failed:
        print(f"\n  WARNINGS: Failed models: {', '.join(failed)}")
        print(f"  These models will need their config fixed before full run.")

    print(f"\n  STATUS: {'READY for full benchmark' if n_ok >= 3 else 'FIX ISSUES above'}")
    print()


if __name__ == "__main__":
    main()
