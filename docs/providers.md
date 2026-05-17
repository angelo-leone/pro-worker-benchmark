# Running the Benchmark Against Any Provider

The runner uses [litellm](https://docs.litellm.ai/) under the hood, so it speaks to any provider that litellm supports. The official v2.0 results were generated against Vultr Serverless Inference. The pipeline is provider-agnostic; only `config.yaml` and your environment variables change.

This guide covers the five most common targets. The default `judges` list in `config.yaml` keeps the same three-judge panel used in the paper; you can leave it untouched even if your candidate model lives on a different provider.

## OpenAI (GPT-4o, GPT-4o-mini, o1, o3, etc.)

```bash
export OPENAI_API_KEY=sk-...
```

`config.yaml`:

```yaml
models:
  - id: "openai/gpt-4o"
    name: "GPT-4o"
    provider: "openai"
  - id: "openai/gpt-4o-mini"
    name: "GPT-4o-mini"
    provider: "openai"
```

Run:

```bash
python -m src.runner --models "openai/gpt-4o"
```

## Anthropic (Claude Opus, Sonnet, Haiku)

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

```yaml
models:
  - id: "anthropic/claude-opus-4-7"
    name: "Claude Opus 4.7"
    provider: "anthropic"
  - id: "anthropic/claude-sonnet-4-6"
    name: "Claude Sonnet 4.6"
    provider: "anthropic"
  - id: "anthropic/claude-haiku-4-5"
    name: "Claude Haiku 4.5"
    provider: "anthropic"
```

```bash
python -m src.runner --models "anthropic/claude-opus-4-7"
```

Notes:

* Anthropic charges separately for prompt caching; PWB does not currently emit `cache_control` markers, so caching is opt-in via your client.
* The `system_prompt.md` content goes into the Anthropic `system` parameter automatically via litellm.

## AWS Bedrock

```bash
export AWS_ACCESS_KEY_ID=AKIA...
export AWS_SECRET_ACCESS_KEY=...
export AWS_REGION_NAME=us-east-1
```

```yaml
models:
  - id: "bedrock/anthropic.claude-opus-4-7-20260101-v1:0"
    name: "Claude Opus 4.7 (Bedrock)"
    provider: "bedrock"
  - id: "bedrock/meta.llama-4-405b-instruct-v1:0"
    name: "Llama 4 405B (Bedrock)"
    provider: "bedrock"
```

Bedrock model IDs change as new models are onboarded; check `aws bedrock list-foundation-models --region <region>` for the current catalog.

## Google Vertex AI (Gemini)

```bash
export VERTEX_PROJECT=your-gcp-project
export VERTEX_LOCATION=us-central1
gcloud auth application-default login
```

```yaml
models:
  - id: "vertex_ai/gemini-2.5-pro"
    name: "Gemini 2.5 Pro"
    provider: "vertex_ai"
  - id: "vertex_ai/gemini-2.5-flash"
    name: "Gemini 2.5 Flash"
    provider: "vertex_ai"
```

## Local Ollama (any local model)

```bash
ollama pull llama3.3:70b
ollama pull qwen3:32b
ollama serve  # if not already running
```

```yaml
models:
  - id: "ollama/llama3.3:70b"
    name: "Llama 3.3 70B (local)"
    provider: "ollama"
    api_base: "http://localhost:11434"
```

```bash
python -m src.runner --models "ollama/llama3.3:70b"
```

For local models the runner ignores rate limits, so set `settings.concurrency` to whatever your GPU memory allows (typically 1 for a single 70B-class model).

## Mixing providers

`config.yaml` accepts a heterogeneous list. The judge panel and candidate models can live on different providers; you only need to set every relevant API key in your environment.

```yaml
models:
  - id: "openai/gpt-4o"
    name: "GPT-4o"
    provider: "openai"
  - id: "anthropic/claude-opus-4-7"
    name: "Claude Opus 4.7"
    provider: "anthropic"
  - id: "vertex_ai/gemini-2.5-pro"
    name: "Gemini 2.5 Pro"
    provider: "vertex_ai"

judges:
  - id: "openai/mistralai/Devstral-2-123B-Instruct-2512"
    name: "Devstral-2 123B (judge)"
    provider: "vultr"
    api_base: "https://api.vultrinference.com/v1"
    temperature: 0.0
  # ... rest of the default panel unchanged
```

## Custom endpoints (vLLM, TGI, OpenRouter, Together, etc.)

Any OpenAI-compatible endpoint works by prefixing the model id with `openai/` and setting `api_base`:

```yaml
models:
  - id: "openai/your-org/your-model"
    name: "Your model"
    provider: "openai"
    api_base: "https://your-vllm.example.com/v1"
```

The runner injects whatever is in `OPENAI_API_KEY` (or the provider-specific key) as the Bearer token. For OpenRouter:

```bash
export OPENROUTER_API_KEY=sk-or-...
```

```yaml
models:
  - id: "openrouter/anthropic/claude-opus-4-7"
    name: "Claude Opus 4.7 (OpenRouter)"
    provider: "openrouter"
```

## Cost considerations

A full v2.0 run on a single candidate model is:

> 320 prompts × 5 runs × 2 variants (baseline, prompted) = 3,200 model calls
> Plus judge calls: 3,200 × 3 judges = 9,600 judge calls

Approximate cost ranges (May 2026 list prices, before any volume discount):

| Provider | Candidate ($/run) | Judges ($/run) | Total per model |
|---|---|---|---|
| OpenAI GPT-4o-mini as candidate | $2 to $4 | $40 to $70 (default Vultr judges) | $42 to $74 |
| OpenAI GPT-4o as candidate | $30 to $60 | $40 to $70 | $70 to $130 |
| Anthropic Claude Opus 4.7 as candidate | $80 to $140 | $40 to $70 | $120 to $210 |
| Local Ollama 70B as candidate | $0 (compute only) | $40 to $70 | $40 to $70 |
| All on Vultr inference | $20 to $30 | $40 to $70 | $60 to $100 |

Multi-turn (Layer 2) is the long tail because each scenario fires 5 sequential calls and accumulates context. The single largest cost driver is judge calls, not candidate calls, because the judges process the full response three times each.

To dry-run cheaply, set `--layers 1` and lower `runs_per_prompt` to 1 in `config.yaml`. Cost falls to roughly 1/15 of the headline number while dimension ordering and approximate ranks remain stable.
