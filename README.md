---
license: cc-by-nc-4.0
base_model: Qwen/Qwen3-4B
base_model_relation: adapter
library_name: peft
pipeline_tag: text-classification
tags:
  - forecasting
  - probabilistic-forecasting
  - calibration
  - uncertainty-quantification
  - regression
  - prediction-markets
  - brier-score
  - lora
  - peft
  - qwen3
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

**The dev comparison carries the claim.** Its Brier half-width is **0.0030**.

**CORRECTED 2026-08-29.** An earlier version of this paragraph called that
"well-powered" on the grounds that the half-width equalled the measured
seed-to-seed noise floor, and concluded the test "resolves as finely as two runs
of the same configuration differ". **That was wrong, and in the unsafe
direction.** ±0.003 is the standard deviation of a *single* run; the difference
between *two* independently trained runs has a standard deviation of about
0.003 × √2 ≈ 0.004, so the 95% threshold is closer to **±0.008**. Measured
directly: three runs of one unchanged recipe gave 0.1752 / 0.1803 / 0.1805.

**The null above survives this and is strengthened by it** — a wider true
uncertainty makes "indistinguishable" easier to support, not harder. But the
half-width is **narrower** than run-to-run variation, not equal to it, so this
comparison could have called a difference significant that was only a different
training run. Full measurement in the
[1.7B card](https://huggingface.co/Laplace-AI-Research/cournot-cold-1-7b).

**The published comparison is consistent but cannot carry the claim alone.** Its
half-width is 1.8× the effect it is measuring; separating the two models there
would need n≈888, roughly 28 more days of accrual. The 4B's nominal lead on that
split is inside its own noise and should not be read as a win.

**So: indistinguishable on dev at n=3,000, consistent on published at n=277.**
Not "the 4B is better."


**The ladder now has a third rung.**
[`Cournot-Cold 1.7B`](https://huggingface.co/Laplace-AI-Research/cournot-cold-1-7b)
extends the same control down to 1.7B, and **it does not tie** — it is
+0.0119 Brier [+0.0080, +0.0158] worse than this model on dev. The degradation is
entirely **resolution**; calibration is flat across the whole 4.7× span. So the
small model learns *how confident to be* as well as the large one, and loses only
*which questions to be confident about*.

**Read its training-variance section before trusting any interval on this page.**
Two runs of an identical recipe differ by more than a question-clustered
bootstrap implies, and a model-vs-model difference below roughly **0.008 Brier**
should be treated as unresolved — which is wider than several intervals quoted
below. The nulls survive that (a wider uncertainty makes "indistinguishable"
easier to support, not harder); any positive claim needs it.


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

**Measured on this model**, not inherited. Scored off-venue on 778 Kalshi
judgment questions and 3,000 Polymarket questions:

| stratum | n | Brier | resolution | BSS |
|---|---:|---:|---:|---:|
| Kalshi, all | 778 | 0.1716 | 0.0395 | +5.7% |
| **Kalshi Politics** | 209 | 0.0788 | **0.1199** | **+56.1%** |
| **Kalshi Elections** | 461 | 0.2201 | **0.0082** | **−19.0%** |
| Polymarket | 3,000 | 0.2004 | 0.0057 | −7.0% |

**The aggregate is a composition artifact and both numbers belong here.** +5.7%
across all of Kalshi is 59% obscure local elections dragging down a stratum that
beats the home venue — Politics discriminates *better* off-venue (0.1199) than
on Manifold dev (0.0708).

The 8B produces 0.1194 and 0.0080 on the same two strata. **The failure mode is
identical at half the parameters.**

1. **It needs the question's subject to have a public reference class.** On the
   461 Kalshi *Elections* questions above this model collapses to resolution
   **0.0082**. Those questions name obscure individuals in local races — *"Will
   Peter Chatzky be the Democratic nominee for NY-17?"* There is no reference
   class for such a name in a text-only prior. **That is a lookup, not a
   forecast, and this model cannot perform it.**
2. **It cannot do mechanical threshold or counting questions.** Questions needing
   a time series and precise arithmetic are outside a judgment prior.
3. **It is weakest on sports**, on a keyword-identified subset that under-recalls.
4. **Short horizons.** Under 7 days it never beats a market crowd at any point.

### Where the two models are compared directly

Paired on the 778 Kalshi questions: Brier **−0.0064 [−0.0129, +0.0000]**,
resolution **+0.0020 [−0.0053, +0.0099]**. Not significant — though the Brier
upper bound sits exactly at zero, so read it as *not separated* rather than as
excluding a 4B advantage.

### The contamination-free Kalshi subset, where this model does not beat a constant

The 778 above include questions that resolved before the training freeze. **117 of
them resolved after it** (`resolved_at > 2026-08-15T00:00:00Z`), and that subset is
contamination-free by the same rule as the published split. It is also the least
flattering number in this card, so it is stated rather than omitted.

| | n | Brier | resolution |
|---|---:|---:|---:|
| Cournot-Cold 4B | 117 | 0.1891 | 0.0213 |
| Cournot-Cold 8B | 117 | 0.2064 | 0.0356 |
| **a constant at the base rate (0.2308)** | 117 | **0.1775** | 0.0000 |

Paired bootstrap, 10,000 draws, clustered on `question_id` (each of the 117 is its
own `event_ticker`, so question-level resampling *is* the cluster bootstrap):

> **4B minus constant: +0.0116 [-0.0280, +0.0501] — not significant.**
> **4B minus 8B: -0.0173 [-0.0356, +0.0016] — not significant.**

**Read this as: on 117 contamination-free out-of-venue questions, this model is
statistically indistinguishable from predicting the base rate on every one.**

And note the shape of the 4B/8B gap here: the 4B has the lower Brier but the
**lower resolution** (0.0213 against 0.0356). Scoring better while discriminating
less is hedging toward the base rate, not superior forecasting — which is why the
Brier column alone must not be read as "the 4B wins on Kalshi".

The subset is small and lopsided — 88 of 117 are Elections, the stratum identified
above as this model's worst — so it is not a verdict on the venue. It says: **the
transfer result above rests partly on pre-freeze questions, and the clean slice of
it does not show skill.**

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
- **Seed:** 20260822, the same seed the 8B used, and **replicated**. A second
  full-corpus run at seed 20260828 differs by **+0.0021 Brier [−0.0009, +0.0052]**
  — not significant, with the interval half-width at the ±0.003 seed-noise floor.
  The 8B replicates equally (+0.0012 [−0.0016, +0.0041]). Neither shipped number
  is one seed's luck.

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

Every number here is recomputed from the shipped forecasts by `verify.py`, in
this repository, with no model and no GPU. That is the check that matters and it
is the one you can run.

Behind it sits an internal decisions log recording how each number was derived
and which results **failed** — including, for this model, a published-split lead
that did not survive its own interval. **That log is not public**, so nothing in
this card depends on it: every claim above is either reproducible from the files
here or stated as unverified.
