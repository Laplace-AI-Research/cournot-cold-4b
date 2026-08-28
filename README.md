---
license: cc-by-nc-4.0
base_model: Qwen/Qwen3-4B
base_model_relation: adapter
library_name: peft
pipeline_tag: text-classification
tags: [forecasting, calibration, prediction-markets, brier-score, lora]
model-index:
  - name: cournot-cold-4b
    results:
      - task: {type: forecasting, name: Binary event forecasting}
        dataset:
          name: Cournot published split (Manifold, resolved after 2026-08-15 freeze)
          type: manifold-published
        metrics:
          - {type: brier, value: 0.1831, name: Brier score}
          - {type: calibration, value: 0.0034, name: Murphy calibration}
          - {type: resolution, value: 0.0669, name: Murphy resolution}
          - {type: ece, value: 0.0526, name: ECE (equal-width, 10 bins)}
---

# Cournot-Cold 4B

A **calibrated probability estimator** for binary questions about future events.
Given a question, its resolution criteria, its scheduled resolution date and an
`as_of` date, it returns a probability in [0, 1]. It conditions on the **question
text alone** — no retrieval, no news, no market price.

LoRA adapter + scalar regression head on `Qwen/Qwen3-4B`.

**Weights:** [`Laplace-AI-Research/cournot-cold-4b`](https://huggingface.co/Laplace-AI-Research/cournot-cold-4b)
— LoRA adapter, 264 MB.
**Evidence:** [`Laplace-AI-Research/cournot-cold-4b`](https://github.com/Laplace-AI-Research/cournot-cold-4b)
— the eval splits behind every claim, this model's raw forecasts, the metric
code, and `verify.py`, which recomputes every number below without a model or a
GPU.

---

## Read this first: this model exists to be compared

**Cournot-Cold 4B and [Cournot-Cold 8B](https://huggingface.co/Laplace-AI-Research/cournot-cold-8b)
are statistically indistinguishable.** They were trained on the same 81,870
questions, with the same targets, the same seed, the same footing, and differ
only in base model size — 4B against 8B.

That is the point of releasing both. A matched-corpus, matched-seed comparison
across a 2× parameter span is a control almost nobody runs, and the result is
the artifact:

| | dev (n=3,000) | published (n=277) |
|---|---|---|
| Brier delta, 4B − 8B | **+0.0010 [−0.0020, +0.0040]** | −0.0062 [−0.0175, +0.0048] |
| resolution delta | −0.0011 [−0.0042, +0.0020] | +0.0042 [−0.0108, +0.0203] |
| verdict | **not significant** | not significant |

Paired, question-clustered bootstraps, 10,000 resamples, read by expectation.

**The dev comparison carries the claim, and it is well-powered**: its Brier
half-width is **0.0030**, exactly the measured seed-to-seed noise floor of this
setup. The test resolves as finely as two runs of the same configuration differ,
which is the most a comparison here can do.

**The published comparison is consistent but cannot carry the claim alone.** Its
half-width is 1.8× the effect it is measuring; separating the two models there
would need n≈888, roughly 28 more days of accrual. The 4B's nominal lead on that
split is inside its own noise and should not be read as a win.

**So: indistinguishable on dev at n=3,000, consistent on published at n=277.**
Not "the 4B is better."

### Why you would take the 4B

Nothing in the numbers. Everything in the deployment:

- **half the base model** — ~8 GB in bf16 against ~16 GB
- **fits a 16 GB card** with real headroom, where the 8B does not
- **264 MB adapter** against 349 MB
- lower latency and roughly half the serving cost per forecast

If your constraint is accuracy, the two are the same model. If your constraint is
anything else, take this one.

---

## The headline number

**Brier 0.1831** on 277 questions that resolved after a freeze date committed in
writing before it passed.

| | published (headline) | dev (iteration only) |
|---|---:|---:|
| n | 277 | 3,000 |
| **Brier** | **0.1831** | 0.1684 |
| base rate | 0.4946 | 0.3970 |
| base-rate Brier (uncertainty) | 0.2500 | 0.2394 |
| Murphy calibration ↓ | 0.0034 | 0.0007 |
| Murphy resolution ↑ | 0.0669 | 0.0708 |
| ECE, equal-width / equal-mass (10 bins) | 0.0526 / 0.0498 | 0.0172 / 0.0161 |
| distinct output values | 78 | 100 |
| forecast sd | 0.253 | 0.281 |
| BSS vs base rate *(diagnostic, not a comparability claim)* | +26.7% | +29.6% |

Reported the way the field's benchmark maintainers ask for it — a raw Brier with
its corpus, base rate and horizon attached, plus the decomposition — rather than
as a skill score against our own baseline.

---

## Contamination

**Zero of the 3,000 dev questions and zero of the 277 published questions
resolved before the base model was released.** Outcome memorisation is closed by
construction, not by argument.

- **Freeze: 2026-08-15**, committed in a dated, public git history *before* it
  passed.
- The split is gated on **`Qwen3-4B`'s public release date (2025-04-29)**, not on
  a stated pretraining cutoff — a release date is externally checkable, a cutoff
  is a vendor claim.

---

## What this model cannot do

The limits below were measured on **Cournot-Cold 8B**, on the same corpus and
the same training recipe. Given the two models are statistically
indistinguishable on both splits, they are the best available estimate for this
one — but they are **inherited, not measured here**, and are flagged as such
throughout.

1. **It needs the question's subject to have a public reference class.** On 461
   Kalshi *Elections* questions the 8B collapses to resolution 0.0080 [0.0044,
   0.0215]. Those questions name obscure individuals in local races. **That is a
   lookup, not a forecast.**
2. **It cannot do mechanical threshold or counting questions.** Questions needing
   a time series and precise arithmetic are outside a judgment prior.
3. **It is weakest on sports**, on a keyword-identified subset that under-recalls.
4. **Short horizons.** Under 7 days it never beats a market crowd at any point.

### Not measured for this model, and deliberately absent

**Transfer has not been tested for the 4B.** The Kalshi and Polymarket forecasts
that support claims 1–3 belong to the 8B, and this repository **does not ship
them** rather than presenting another model's results as its own. `verify.py`
says so where the transfer section would be.

If off-venue behaviour matters to your use, use the 8B, whose transfer results
are measured and published.

---

## Evaluation data

- **`published`** (headline, n=277) — Manifold questions resolving after
  2026-08-15. Never trained on. The only source of an external number.
- **`dev`** (n=3,000) — resolving 2025-08-15 to 2026-08-15. Gates iteration,
  **never published as a headline claim.**

All intervals are paired, question-clustered bootstraps (10,000 resamples).
Seed-to-seed noise on this setup is **±0.003 Brier**, so smaller differences are
not findings.

---

## Training

- **Base:** `Qwen/Qwen3-4B` (Apache 2.0), LoRA r=32 α=64 dropout=0.05,
  `all-linear`, plus a trainable scalar head (`modules_to_save=["score"]`).
- **Objective:** Brier/MSE loss on a scalar output against the **terminal 0/1
  outcome**.
- **Corpus:** 81,870 resolved Manifold questions, all resolving before
  2025-08-14 — identical to the 8B's.
- **Footing:** chat template on **both** train and score. Load-bearing; a
  mismatch here silently invalidated an earlier result on this project.
- **Padding:** right. The head pools the last non-pad token.
- **Seed:** 20260822, the same seed the 8B used. **One seed only** — neither
  model is seed-replicated at full corpus, so the comparison between them is
  matched, but neither absolute number is replicated.

**Reproduction note:** `transformers >= 4.51.0` is required for
`Qwen3ForSequenceClassification`; `config.pad_token_id` must be set explicitly.

---

## Training data provenance and licensing

The corpus is derived from the Manifold Markets public API.

**Manifold's terms restrict bulk API data to personal and non-commercial use, and
state that it may not be used to train machine learning models for commercial
purposes without a data licence** (data@manifold.markets). This adapter is
released under **CC BY-NC 4.0** on that basis. Anyone intending commercial use
must obtain that licence from Manifold independently; it is not ours to grant.

The raw corpus is **not redistributed**. `DATASHEET.md` describes its
composition, collection and known defects in its place.

The base model's own licence (Apache 2.0) is unaffected.

---

## Reproducing

Every headline number is recomputed from the shipped forecasts by `verify.py` —
**no model, no GPU, no network**:

```bash
uv run python verify.py
```

To regenerate those forecasts from the weights instead:

```bash
uv run python scripts/scalar_score.py \
  --adapter Laplace-AI-Research/cournot-cold-4b \
  --base-model Qwen/Qwen3-4B \
  --set eval/published_eval.json \
  --out published.jsonl \
  --chat-template
```

---

## Intended use

A **prior**, not an oracle. Use it for base rates, cold-start estimates, and
questions with no market. **Never** as the sole input to a consequential
decision.

---

## Provenance of claims

Every number here is derived in `docs/10-decisions.md` of the research
repository, which also records the results that **failed** — including, for this
model specifically, a published-split lead that did not survive its own interval
(entry 2026-08-27aj).
