---
layout: post
title: "Train Your First LLM on a Free TPU"
date: 2026-07-18
description: "Hands-on: train a ~10M-parameter GPT end-to-end on a free Colab/Kaggle TPU with JAX + Flax NNX, using the tpu-llm-from-scratch repo. Real commands, honest gotchas, and the path from free tier to TPU pods."
tags: [tpu, jax, flax-nnx, llm-training, beginner, colab]
toc: true
math: false
---

# Train Your First LLM on a Free TPU

Here's a fact that still surprises people: the same chip family Google uses to train Gemini is sitting in a free Colab notebook, waiting for you, right now. Eight TPU cores. No credit card.

Most people never touch them, because the TPU ecosystem has a reputation: *that's for Google-scale infra teams.* This post is the counterexample. In the next half hour you'll train a small GPT — real transformer, real training loop, real generated text — on a free TPU, using the [`tpu-llm-from-scratch`](https://github.com/keshan/tpu-llm-from-scratch/) repo. Every line of the code is meant to be read; nothing is hidden behind a framework.

## What you're going to build

A character-level GPT (~10M parameters, the `tiny` preset) trained on Tiny Shakespeare. Yes, it's small. That's the point:

- **Small enough** to train end-to-end in minutes on a free TPU, so the feedback loop is tight enough to actually learn from.
- **Structurally identical** to the big ones: token embeddings, causal self-attention, MLP blocks, residual stream, AdamW with a cosine schedule. When you later read a Llama or Gemma paper, you'll recognize every part.
- **Written in JAX + Flax NNX**, the stack that's native to the TPU. If you know PyTorch and are JAX-curious.

At the end, your model will produce text like this — not good Shakespeare, but unmistakably *trying*:

```
DUKE VINCENTIO:
What say you so? the wilt of my sun,
That we shall the good for the still...
```

Ten million parameters, a few minutes of training, and it has learned English spelling, line structure, character names, and stage directions from nothing but "predict the next character." That moment — watching structure emerge from a loss curve — is the whole reason to do this hands-on rather than read about it.

## Step 0 — Get a TPU (three free doors)

| Option | What you get | Catch |
|--------|-------------|-------|
| **Google Colab** | v2-8/v3-8-class runtime, zero setup | Sessions time out; availability varies |
| **Kaggle** | TPU VM with weekly quota hours | Quota, but generous for this project |
| **TRC (TPU Research Cloud)** | Real TPU VMs, free for research/education | Short application; the best-kept secret in this field — as a learner or content creator you likely qualify |

For today, Colab is fine: **Runtime → Change runtime type → TPU**. (The full comparison, including paid options for later, is in the [TPU access guide](https://github.com/keshan/resources/tpu-access-guide.md).)

## Step 1 — Clone, install, verify

```bash
git clone https://github.com/keshan/tpu-llm-from-scratch
cd tpu-llm-from-scratch
pip install -r requirements.txt
```

Then the single most important cell in any TPU notebook:

```python
import jax
jax.devices()
# [TpuDevice(id=0, ...), TpuDevice(id=1, ...), ... 8 entries]
```

**Do not skip this.** If you see `CpuDevice`, you're about to spend twenty minutes "training on TPU" on a CPU — the classic first-timer trap. The [colab quickstart](https://github.com/keshan/tpu-llm-from-scratch/notebooks/colab_quickstart.ipynb) has the fix (install the TPU-enabled jaxlib, restart the runtime).

Eight devices, one chip package or four — this is your first contact with the idea from Part 1 that a "TPU" is really a *mesh of cores* that JAX addresses individually. Today they'll do data parallelism; in [`scaling.md`](https://github.com/keshan/tpu-llm-from-scratch/scaling.md) the same mesh does FSDP and tensor parallelism.

## Step 2 — Train

```bash
python train.py --preset tiny
```

That's it. You'll watch the loss fall from ~4.3 (uniform guessing over the character vocabulary) toward ~1.5. Three things worth *noticing* while it runs, because they're the deep content of this series in miniature:

**1. The first step is slow. Every other step is fast.** That's XLA compiling your whole training step into one fused TPU program, then replaying it thousands of times. This is the JAX contract — trace once, compile, iterate — and it's why the code has no explicit device loops.

**2. Nothing in `model.py` mentions the TPU.** The model is pure math on arrays. All hardware awareness lives in one function in `train.py` (`make_mesh_and_shardings`) that lays data out across the 8 cores. Decoupling *what you compute* from *where it runs* is the core JAX/XLA design idea, and it's what makes the free→paid scaling story real.

**3. The whole config fits on one screen.** Open [`config.py`](https://github.com/keshan/tpu-llm-from-scratch/config.py): `n_layer`, `n_head`, `n_embd`, `block_size`, learning rate, warmup. There is no knob you can't see. When a run misbehaves, you can reason about it.

## Step 3 — Sample

```bash
python sample.py --preset tiny --prompt "ROMEO:" --max_new_tokens 300 --temperature 0.8
```

Play with `--temperature` (0.3 = repetitive and safe, 1.2 = chaotic) and `--top_k`. You now have an intuition for the two sampling knobs every LLM API exposes — because you turned them on a model whose every weight you trained yourself.

## Honest section: where it will hurt

I promised this series wouldn't do marketing, so:

- **Colab TPUs come and go.** Peak hours can mean no TPU available. Kaggle's quota is more predictable; TRC is the real fix.
- **Preemption is real.** Free runtimes die. The repo checkpoints with Orbax so `train.py` can resume — get in the habit of saving to Drive (quickstart Cell 7).
- **The error messages are XLA's, not Python's.** A shape mismatch surfaces as a compiler complaint. The debugging rhythm (`jax.debug.print`, running on CPU first) takes a week to internalize. It's a real tax; you pay it once.

## Where this goes next

You've now done, at toy scale, exactly what a frontier lab does at pod scale. 

- The `tiny` preset → `small`, `scaled`, and the multi-host story in [`scaling.md`](https://github.com/keshan/tpu-llm-from-scratch/scaling.md); graduation to [MaxText](https://maxtext.readthedocs.io/) when you want production scale.
## Key takeaways

> - A free Colab/Kaggle TPU gives you 8 real cores — enough to train a ~10M-param GPT end-to-end in minutes.
> - `jax.devices()` before anything else. Ever.
> - The JAX contract: pure functions, trace-and-compile (slow first step, fast forever after), and hardware layout isolated to one mesh function — that separation is what lets the identical code scale to pods.
> - Small model, full anatomy: everything you just trained recurs in every frontier LLM.

## Further reading

- [`tpu-llm-from-scratch`](https://github.com/keshan/tpu-llm-from-scratch/) — the repo, including the [cell-by-cell Colab walkthrough](https://github.com/keshan/tpu-llm-from-scratch/notebooks/colab_quickstart.md)
- [TPU access guide](https://github.com/keshan/resources/tpu-access-guide.md) — Colab vs Kaggle vs TRC vs paid, with gotchas
- [TPU Research Cloud](https://sites.research.google/trc/about/) — apply; it's genuinely free
- [JAX AI Stack: LLM pretraining tutorial](https://docs.jaxstack.ai/en/latest/JAX_for_LLM_pretraining.html) — the official companion path
