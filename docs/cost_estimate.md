# Benchmark Cost Estimate & Provider Split

## Budget: $200 (DigitalOcean) + $305 (Vultr) = $505 total

## Token Estimates Per API Call
- **Model candidate call**: ~800 input tokens, ~1,000 output tokens
- **Judge call**: ~3,000 input tokens (rubric + few-shot + prompt + response), ~150 output tokens

## Provider Pricing Comparison

| Model | DO Input/1M | DO Output/1M | Vultr Input/1M | Vultr Output/1M |
|-------|-------------|--------------|----------------|-----------------|
| GPT-4o | $2.50 | $10.00 | N/A | N/A |
| Claude Sonnet 4.5 | $3.00 | $15.00 | N/A | N/A |
| Claude Haiku 4.5 | $1.00 | $5.00 | N/A | N/A |
| GPT-4o-mini | $0.15 | $0.60 | N/A | N/A |
| Llama 3.3 70B | $0.65 | $0.65 | N/A | N/A |
| Qwen3-32B | $0.25 | $0.55 | N/A | N/A |
| DeepSeek R1 70B | $0.99 | $0.99 | N/A | N/A |
| Nemotron-3-120B | $0.30 | $0.65 | N/A | N/A |
| All Vultr models | N/A | N/A | $0.55 | $2.75 |

**Key insight**: DO's self-hosted open models are CHEAPER than Vultr for output-heavy
calls (model responses). Use DO for most work; use Vultr for models DO doesn't have.

---

## Recommended Configuration (runs_per_prompt=5)

### Calls Per Model (both conditions)
- Layer 1: 200 prompts × 5 runs × 2 = 2,000 calls
- Layer 2: 16 scenarios × 5 turns × 2 = 160 calls
- Layer 3: 40 prompts × 5 runs × 2 = 400 calls
- **Total per model: ~2,560 model calls**

### Judge Calls (3 judges, each scores all responses)
- Total responses across 7 DO models: 7 × 2,560 = 17,920
- Each judge scores all: 17,920 calls per judge
- 3 judges: 53,760 judge calls total

---

## DigitalOcean Budget ($200)

### Candidate Models (7 models)
| Model | Cost/call | Total Calls | Cost |
|-------|-----------|-------------|------|
| GPT-4o | $0.0120 | 2,560 | $30.72 |
| Claude Sonnet 4.5 | $0.0174 | 2,560 | $44.54 |
| Claude Haiku 4.5 | $0.0058 | 2,560 | $14.85 |
| GPT-4o-mini | $0.0007 | 2,560 | $1.84 |
| Llama 3.3 70B | $0.0012 | 2,560 | $3.00 |
| Qwen3-32B | $0.0008 | 2,560 | $1.92 |
| DeepSeek R1 70B | $0.0018 | 2,560 | $4.56 |
| **Candidate subtotal** | | | **$101.43** |

### Judge Panel (3 judges, all on DO for cheapness)
| Judge Model | Cost/call | Total Calls | Cost |
|-------------|-----------|-------------|------|
| GPT-4o-mini | $0.0005 | 17,920 | $9.68 |
| Qwen3-32B | $0.0008 | 17,920 | $14.93 |
| Nemotron-3-120B | $0.0010 | 17,920 | $17.88 |
| **Judge subtotal** | | | **$42.49** |

### **DO Total: ~$144** (leaves ~$56 buffer for retries/overages)

---

## Vultr Budget ($305)

Use for additional model diversity (models not available on DO).

### Additional Candidate Models
| Model | Cost/call | Total Calls | Cost |
|-------|-----------|-------------|------|
| Mistral (7B or 24B) | $0.0032 | 2,560 | $8.17 |
| Llama 3.1 70B | $0.0032 | 2,560 | $8.17 |
| **Vultr candidate subtotal** | | | **$16.34** |

### Cross-Infrastructure Validation
Run 2-3 DO open-weight models also on Vultr to verify scores
are provider-independent (important for reproducibility):
| Model | Cost/call | Total Calls | Cost |
|-------|-----------|-------------|------|
| Llama (cross-check) | $0.0032 | 2,560 | $8.17 |
| **Validation subtotal** | | | **$8.17** |

### Extra Statistical Power
Run 5 additional runs (total 10) on select models for variance analysis:
| Use | Estimated Cost |
|-----|---------------|
| 5 extra runs × 3 models | ~$25 |

### **Vultr Total: ~$50** (leaves ~$255 buffer)

---

## Combined Summary

| Component | Provider | Cost |
|-----------|----------|------|
| 7 candidate models (5 runs each) | DO | $101 |
| 3 judge models (scoring all) | DO | $42 |
| 2 additional candidates (Vultr-only models) | Vultr | $16 |
| Cross-infrastructure validation | Vultr | $8 |
| Extra runs for statistical power | Vultr | $25 |
| **Total estimated** | | **$192** |
| **Budget remaining** | | **$313** |

The large remaining buffer accommodates:
- Rate limit retries (can double call counts in worst case)
- Increasing runs_per_prompt to 10 for publication ($~100 additional)
- Adding more models if needed
- Re-running dimensions that show low reliability

---

## Final Model Lineup (9 models)

| # | Model | Family | Size | Provider | Role |
|---|-------|--------|------|----------|------|
| 1 | GPT-4o | OpenAI | Large | DO | Proprietary flagship |
| 2 | Claude Sonnet 4.5 | Anthropic | Large | DO | Proprietary flagship |
| 3 | Llama 3.3 70B | Meta | 70B | DO | Open-weight large |
| 4 | Qwen3-32B | Alibaba | 32B | DO | Open-weight medium |
| 5 | DeepSeek R1 Distill 70B | DeepSeek | 70B | DO | Reasoning-optimized |
| 6 | GPT-4o-mini | OpenAI | Small | DO | Proprietary small |
| 7 | Claude Haiku 4.5 | Anthropic | Small | DO | Proprietary small |
| 8 | Mistral 24B | Mistral | 24B | Vultr | Open-weight (new family) |
| 9 | Llama 3.1 70B | Meta | 70B | Vultr | Version comparison |
