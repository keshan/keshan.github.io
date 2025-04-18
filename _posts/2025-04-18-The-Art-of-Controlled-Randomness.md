---
layout: post
title: "The Art of Controlled Randomness: A Deep Dive into Sampling techniques in LLMs"
date: 2025-04-18
description: "A Deep Dive into Sampling Techniques in LLMs"
img: controlled_randomness.jpg
fig-caption: # Add figcaption (optional)
tags: [JAX, Flax, LLM, Generation, Sampling]
---

Large Language Models (LLMs) are powerful predictors. Given a sequence of text, they excel at calculating the likelihood of *every possible next token* (or word) in their vast vocabulary. These likelihoods start as raw scores called **logits**. But how do we turn a bunch of potential next words into a coherent, creative, and controlled stream of text?

Simply picking the word with the absolute highest logit (greedy decoding) often leads to repetitive and dull output. Also, picking a random word from the vocabulary (naive random sampling) leads to gibberish. We need smarter **sampling strategies** that introduce controlled randomness, balancing diversity and faithfulness.

This tutorial dives deep into the most common sampling techniques – **Temperature Scaling**, **Top-K Filtering**, **Top-P (Nucleus) Sampling**, and **Min-P Filtering**. We'll explore:

1.  **How** each technique manipulates the logits.
2.  **Why** each technique is useful (the problem it solves).
3.  **How they work together** in a specific order.
4.  **How to implement and use them** with JAX.

Let's dive in!

Imagine our LLM has processed the input `"The rocket lifted off towards the"` and needs to predict the next word. It outputs logits (raw, unnormalized scores) for its entire vocabulary. Let's focus on a few plausible candidates (here the logits values are just made up for illustration):

*   `moon`: logit = `3.5`
*   `stars`: logit = `3.1`
*   `sky`: logit = `2.9`
*   `station`: logit = `2.0`
*   `launchpad`: logit = `-0.5` (less likely, but possible)
*   ...(thousands of other words with much lower logits)

Our goal is to use sampling techniques to intelligently select *one* of these words.

**1. Adjusting the "Confidence" in other words, Temperature Scaling**

*   **Why?** Raw logits can sometimes be extremely "peaky," with one token having a vastly higher score than others. This leads back towards greedy, less creative output. Temperature lets us smooth out or sharpen this distribution *before* filtering.
*   **How?** We divide every logit by a `temperature` value.
    *   `T > 1.0`: Decreases the differences between logits, making probabilities more uniform (flatter). Increases randomness, allowing less likely words a better chance. Think of it like turning *up* the creative chaos.
    *   `T < 1.0`: Increases the differences, making high-probability words even more likely (peakier). Reduces randomness, focusing generation. Think of it like turning *down* the chaos for more focused output.
    *   `T = 1.0`: No change.
*   **Implementation of `temperature_scale` in JAX:**

    ```python
    # Constants for numerical stability
    EPSILON = 1e-9

    def temperature_scale(logits: jnp.ndarray, temperature: float) -> jnp.ndarray:
        safe_temperature = max(temperature, EPSILON) # Avoid division by zero
        return logits / safe_temperature
    ```
    Simple division, but crucial for reshaping the landscape.

*   **Example (Temperature = 0.8):** for an interactive demo, check out <a href="https://blog.keshan.dev/llm-demo-apps/" target="_blank">here</a>.
    *   `moon`: 3.5 / 0.8 = `4.375`
    *   `stars`: 3.1 / 0.8 = `3.875`
    *   `sky`: 2.9 / 0.8 = `3.625`
    *   `station`: 2.0 / 0.8 = `2.5`
    *   `launchpad`: -0.5 / 0.8 = `-0.625`

    Notice the gaps between the top scores are now larger relative to their magnitude, making "moon" even more dominant after this step. If T > 1.0, the opposite would happen.

**2. Pruning the long tail, in other words Min-P Filtering**

*   **Why?** Sometimes, even after temperature scaling, there are many tokens with non-negligible but still very low probabilities compared to the best option. Min-P offers a dynamic way to filter these out based on the *peak* probability in the current distribution. It helps remove the long tail without needing a fixed `k` or `p`.
*   **How?**
    1.  Calculate the probability of each token using softmax on the (potentially temperature-scaled) logits.
    2.  Find the maximum probability (`max_prob`).
    3.  Set a threshold: `min_threshold = max_prob * min_p`.
    4.  Discard (set logits to `-inf`) any token whose probability is *less than* `min_threshold`, *unless* it's one of the tokens that had the `max_prob` (to ensure we always keep the most likely option).
*   **JAX Implementation (`min_p_logits`):**

    ```python
    def min_p_logits(logits: jnp.ndarray, p: float) -> jnp.ndarray:
        probs = nnx.softmax(logits, axis=-1) # Convert current logits to probs
        max_prob = jnp.max(probs, axis=-1, keepdims=True)
        threshold = max_prob * p

        # Identify indices corresponding to max probability
        max_prob_indices = probs >= (max_prob - EPSILON)

        # Keep max prob tokens and tokens above the threshold
        mask_below_threshold = probs < threshold
        # Mask is True for tokens we want to discard
        mask = jnp.where(max_prob_indices, False, mask_below_threshold)

        # Apply the mask (set discarded logits to -inf)
        return jnp.where(mask, -jnp.inf, logits)
    ```

*   **Example (Continuing with T=0.8 logits, Min-P = 0.1):** for an interactive demo, check out [here](https://blog.keshan.dev/llm-demo-apps/).
    1.  **Softmax:** Convert `[4.375, 3.875, 3.625, 2.5, -0.625, ...]` to probabilities. Let's approximate: `exp(4.375) ≈ 79.4`, `exp(3.875) ≈ 48.2`, `exp(3.625) ≈ 37.5`, `exp(2.5) ≈ 12.2`, `exp(-0.625) ≈ 0.5`. Sum ≈ `177.8` (just for these 5).
        *   `Prob(moon)` ≈ 79.4 / 177.8 ≈ `0.447` (This is `max_prob`)
        *   `Prob(stars)` ≈ 48.2 / 177.8 ≈ `0.271`
        *   `Prob(sky)` ≈ 37.5 / 177.8 ≈ `0.211`
        *   `Prob(station)` ≈ 12.2 / 177.8 ≈ `0.069`
        *   `Prob(launchpad)` ≈ 0.5 / 177.8 ≈ `0.003`
    2.  **Threshold:** `min_threshold = 0.447 * 0.1 = 0.0447`
    3.  **Filter:**
        *   Keep `moon` (prob 0.447 >= 0.0447) -> logit remains `4.375`
        *   Keep `stars` (prob 0.271 >= 0.0447) -> logit remains `3.875`
        *   Keep `sky` (prob 0.211 >= 0.0447) -> logit remains `3.625`
        *   Keep `station` (prob 0.069 >= 0.0447) -> logit remains `2.5`
        *   Discard `launchpad` (prob 0.003 < 0.0447) -> logit becomes `-inf`
    *   Our logits are now: `[4.375, 3.875, 3.625, 2.5, -inf, ...]`

**3. Top-K Filtering or The VIP List**

*   **Why?** To impose a hard limit on the number of choices, regardless of their probabilities. This prevents the model from considering truly bizarre (but maybe slightly probable after temperature/min-p) tokens. It ensures a minimum level of focus.
*   **How?** Simply select the `k` tokens with the highest *current* logits and discard all others by setting their logits to `-inf`.
*   **JAX Implementation (`top_k_logits`):**

    ```python
    def top_k_logits(logits: jnp.ndarray, k: int) -> jnp.ndarray:
        # ... (error checks, handle k > vocab_size) ...
        k = min(k, logits.shape[-1])

        # Efficiently find the value of the k-th largest logit
        top_k_values = jax.lax.top_k(logits, k=k)[0] # Gets values, not indices
        kth_value = top_k_values[..., -1:] # The smallest value in the top-k set

        # Create a mask: True for logits >= k-th value
        mask = logits >= kth_value

        # Apply mask: Keep top-k, set others to -inf
        return jnp.where(mask, logits, -jnp.inf)
    ```
    `jax.lax.top_k` is efficient for finding the threshold value.

*   **Example (Continuing, Top-K = 3):** for an interactive demo, check out <a href="https://blog.keshan.dev/llm-demo-apps/" target="_blank">here</a>.
    *   Current logits: `[4.375, 3.875, 3.625, 2.5, -inf, ...]`
    *   The top 3 logits are `4.375` (moon), `3.875` (stars), `3.625` (sky).
    *   The 3rd highest logit is `3.625`.
    *   Filter:
        *   Keep `moon` (4.375 >= 3.625) -> logit `4.375`
        *   Keep `stars` (3.875 >= 3.625) -> logit `3.875`
        *   Keep `sky` (3.625 >= 3.625) -> logit `3.625`
        *   Discard `station` (2.5 < 3.625) -> logit becomes `-inf`
        *   `launchpad` remains `-inf`.
    *   Our logits are now: `[4.375, 3.875, 3.625, -inf, -inf, ...]`

**4. Top-P (Nucleus) Filtering: The Probability Budget**

*   **Why?** Top-K uses a fixed number (`k`), but sometimes the probability distribution is very sharp (only 1-2 good options), and sometimes it's flat (many decent options). Top-P adapts dynamically. It selects the *smallest set* of tokens whose cumulative probability mass exceeds a threshold `p`. This captures the "nucleus" of likely candidates.
*   **How?**
    1.  Convert the *current* logits (after Temp, Min-P, Top-K) into probabilities using softmax.
    2.  Sort these probabilities in descending order.
    3.  Calculate the cumulative sum of the sorted probabilities.
    4.  Find the tokens whose cumulative probability is `<= p`. Crucially, *always include at least the highest probability token*.
    5.  Discard all other tokens by setting their *original* logits (from the input to this function) to `-inf`.
*   **JAX Implementation (`top_p_logits`):**

    ```python
    def top_p_logits(logits: jnp.ndarray, p: float) -> jnp.ndarray:
        # ... (error checks, handle p=1) ...
        probs = nnx.softmax(logits, axis=-1) # Probs from current logits

        # Sort probabilities DESCENDING
        sorted_probs = jnp.sort(probs, axis=-1)[..., ::-1]
        # Get corresponding indices (needed to map back later, though not explicit here)
        # sorted_indices = jnp.argsort(probs, axis=-1)[..., ::-1]

        cumulative_probs = jnp.cumsum(sorted_probs, axis=-1)

        # Create a mask on the *sorted* probabilities
        sorted_mask = cumulative_probs <= p
        # Ensure the top-1 token is always included
        sorted_mask = sorted_mask.at[..., 0].set(True)

        # Find the minimum probability value *within* the nucleus (in the sorted list)
        threshold = jnp.min(
            jnp.where(sorted_mask, sorted_probs, jnp.ones_like(sorted_probs)), axis=-1, keepdims=True
        )

        # Apply this threshold to the *original* probability distribution
        mask = probs >= threshold

        # Apply the final mask to the *input logits*
        return jnp.where(mask, logits, -jnp.inf)
    ```
    This implementation cleverly finds the probability threshold from the sorted list and applies it back to the original probabilities to create the final mask.

*   **Example (Continuing, Top-P = 0.7):** for an interactive demo, check out <a href="https://blog.keshan.dev/llm-demo-apps/" target="_blank">here</a>.
    1.  **Softmax on current logits:** `[4.375, 3.875, 3.625, -inf, -inf, ...]`.
        `exp(4.375) ≈ 79.4`, `exp(3.875) ≈ 48.2`, `exp(3.625) ≈ 37.5`. Others are 0.
        Sum ≈ `165.1`.
        *   `Prob(moon)` ≈ 79.4 / 165.1 ≈ `0.481`
        *   `Prob(stars)` ≈ 48.2 / 165.1 ≈ `0.292`
        *   `Prob(sky)` ≈ 37.5 / 165.1 ≈ `0.227`
    2.  **Sort & Cumulate:**
        *   Sorted Probs: `[0.481 (moon), 0.292 (stars), 0.227 (sky)]`
        *   Cumulative Probs: `[0.481, 0.773, 1.0]`
    3.  **Filter (`p=0.7`):**
        *   Keep `moon` (cumulative 0.481 <= 0.7).
        *   Stop: `stars` (cumulative 0.773 > 0.7). We only keep the tokens *before* crossing the threshold `p`, but always ensure the first one is kept. In this case, only "moon" makes the cut based on `cumulative_probs <= p`.
    4.  **Final Logits:** Apply the mask to the logits we fed *into* this step: `[4.375, -inf, -inf, -inf, -inf, ...]`.


**Orchestrating the Sampling: The `sample_logits` Function**

This function brings everything together. It defines the **order of operations**, which is crucial.

```python
def sample_logits(
    logits: jnp.ndarray,
    rng_key: jax.Array,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    min_p: float | None = None,
    do_sample: bool = True,
) -> jnp.ndarray:
    if not do_sample: # Handle greedy decoding
        return jnp.argmax(logits, axis=-1)

    # 1. Apply temperature scaling
    scaled_logits = temperature_scale(logits, temperature) # Use safe scaling

    logits_for_fallback = scaled_logits # Store for safety

    # 2. Apply filtering (Order: Min-P -> Top-K -> Top-P)
    filtered_logits = scaled_logits
    if min_p is not None and 0 < min_p < 1.0:
        filtered_logits = min_p_logits(filtered_logits, min_p)
    if top_k is not None and top_k > 0:
        filtered_logits = top_k_logits(filtered_logits, top_k)
    if top_p is not None and 0 < top_p < 1.0:
        filtered_logits = top_p_logits(filtered_logits, top_p)

    # 3. Handle edge case: If all logits became -inf (over-filtering)
    all_filtered_infinite = jnp.all(filtered_logits == -jnp.inf, axis=-1, keepdims=True)
    # Fallback to the pre-filtering (but post-temperature) logits if needed
    final_logits_for_sampling = jnp.where(
        all_filtered_infinite,
        logits_for_fallback,
        filtered_logits,
    )

    # 4. Sample from the final distribution
    sampled_indices = jax.random.categorical(rng_key, final_logits_for_sampling, axis=-1)

    return sampled_indices
```

**Key Takeaways:**

1.  **Order:** Temperature -> Min-P -> Top-K -> Top-P. This specific order applies the broad temperature adjustment first, then prunes the dynamic low-end (Min-P), then enforces a hard count limit (Top-K), and finally applies the dynamic probability mass limit (Top-P). Other orderings are possible but would yield different results.
2.  **Fallback:** It includes crucial logic (`jnp.where(all_filtered_infinite, ...)`) to prevent errors if the combination of filters accidentally removes *all* possible tokens. In that rare case, it falls back to sampling from the distribution *after* temperature scaling but *before* any filtering.
3.  **Final Sampling:** `jax.random.categorical` performs the actual sampling. It takes the final, filtered logits, implicitly converts them to probabilities via softmax (since `categorical` works on logits), and draws a sample according to those probabilities using the provided JAX `rng_key`.
4.  **Greedy Option:** If `do_sample=False`, it bypasses all sampling logic and simply returns the index of the highest original logit (`jnp.argmax`).

**Putting it into Practice: Autoregressive Generation**

The `GenerationMixin` class wraps this logic into a usable generation loop.

```python
class GenerationMixin:
    # Core logic using lax.scan
    def _generate_scan_logic(self, ...):
        # ... setup initial state (output_ids, rng, finished flags) ...

        def scan_step(carry, _):
            # ... get current state ...
            # Call the model (self) to get logits for the *next* token
            logits = self(input_ids=current_output_ids, ...)
            next_token_logits = logits[:, current_length - 1, :] # Get the logits for the token we need to predict

            # *** THE KEY CALL ***
            next_token = sample_logits(
                logits=next_token_logits,
                rng_key=sampling_rng,
                temperature=temperature, 
                top_k=top_k, 
                top_p=top_p, 
                min_p=min_p, 
                do_sample=do_sample
            )

            # ... update output_ids, finished flags, rng, length ...
            return next_carry, None

        # ... run jax.lax.scan ...
        final_output_ids = # result of scan
        return final_output_ids

    # Jitted version (using partial for static args)
    _generate_compiled = partial(jax.jit, static_argnames=(...))(_generate_scan_logic)

    # Public API method
    def generate(self, input_ids, ..., use_jit=False):
        # ... input validation, handle RNG, resolve pad/eos tokens ...

        # Decide whether to call the raw Python loop or the JIT-compiled one
        if use_jit:
            final_output_ids = self._generate_compiled(...)
        else:
            final_output_ids = self._generate_scan_logic(...)

        return final_output_ids
```

You can find the full code [here](https://github.com/keshan/jax-layers/blob/main/jaxgarden/models/generation_utils.py).

**Key Aspects of `GenerationMixin`:**

1.  **Autoregression:** It works step-by-step. In each step (`scan_step`), it calls the underlying LLM (`self(...)`) with the *current* sequence to get logits for the *next* token.
2.  **`lax.scan`:** This JAX primitive is used for efficient looping on accelerators (GPU/TPU). It compiles the `scan_step` function and executes it repeatedly.
3.  **Integration:** It seamlessly integrates the `sample_logits` function, feeding it the relevant logits and sampling parameters at each step.
4.  **State Management:** It handles updating the generated sequence (`output_ids`), managing the JAX PRNG key (`rng`), tracking finished sequences (`finished`), and padding.
5.  **JIT Compilation:** It provides an option (`use_jit=True`) to call a `jax.jit`-compiled version (`_generate_compiled`) of the generation loop. This significantly speeds up generation by compiling the Python logic into optimized XLA code, but requires parameters like `temperature`, `top_k`, `top_p`, `min_p`, `do_sample`, `max_length` etc., to be *static* (known at compile time).

**Conclusion: Your Control Panel for AI Creativity**

You now have a deep understanding of how Temperature, Min-P, Top-K, and Top-P sampling work together to shape the output of language models within a JAX framework.

*   **Temperature:** Controls the overall randomness/conservatism.
*   **Min-P:** Dynamically removes the lowest probability tail based on the peak.
*   **Top-K:** Enforces a hard limit on the number of candidates.
*   **Top-P:** Enforces a dynamic limit based on cumulative probability mass.

The `sample_logits` function orchestrates these steps in a specific order, and the `GenerationMixin` integrates this into an efficient, JIT-compilable autoregressive loop. By mastering these parameters, you gain control over your LLM's voice, balancing focus and creativity. Experiment with these tools to unlock new possibilities!