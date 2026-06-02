# TailBlend: Predictor-Aware KV Preemption for Best-of-N LLM Serving in vLLM

## 1. Introduction

Best-of-N (BoN) sampling has become an important inference-time compute pattern for improving large language model (LLM) answer quality. Instead of relying on a single stochastic generation, BoN launches multiple independent candidate branches for the same prompt and selects the best response using a verifier, a reward model, or an external scoring rule. Recent empirical work shows that repeated sampling can scale task coverage substantially as the number of samples increases, especially in verifiable domains such as coding and mathematical reasoning.

From a serving-system perspective, however, BoN turns one user request into a large intra-request parallel workload. A single logical request with `N=16` or `N=32` becomes a group of many decode streams that share the same prompt prefix but diverge in output tokens. This stresses vLLM in a way that ordinary request-level batching does not: the scheduler must keep many sibling branches alive, allocate KV-cache blocks for their generated tokens, and eventually return the user-visible result only when the logical BoN parent is complete.

Existing LLM serving systems are optimized primarily for batching independent requests. In contrast, BoN creates many sibling branches inside one logical request, and those branches compete for the same KV-cache budget while sharing a parent-level completion objective. Under high `N`, the dominant failure mode is often not ordinary batching inefficiency but KV allocation failure: vLLM must preempt a running branch, discard computed KV state, and later recompute lost work. Once this happens, the serving system does not merely "slow down"; it loses work and may delay the completion of the entire BoN parent.

This proposal therefore focuses on the vLLM-side bottleneck: KV-cache pressure and preemption victim selection under BoN serving. Rather than changing the semantics of BoN or delaying branch admission by default, we keep vLLM's eager BoN behavior and intervene only when vLLM already needs to preempt a running branch. The central question is:

> When KV allocation fails under BoN serving, can we choose a better preemption victim than vLLM's default policy by using online sibling-length and parent-completion signals?

We propose **TailBlend**, a predictor-aware preemption policy for vLLM BoN serving. TailBlend uses online observations from completed sibling branches to estimate the remaining lifetime of live branches. It then selects preemption victims by balancing predicted remaining work, already generated output, recompute cost, and the branch's contribution to completing its logical BoN parent. The policy is deliberately narrow: it does not throttle waiting admission, does not reduce decode width under normal execution, and does not route entire datasets or requests. It only replaces the victim scoring rule at the moment of unavoidable KV-cache pressure.

Our current results suggest that TailBlend is a strong main policy for output-tail dominated BoN workloads. On GSM8K and MBPP, it improves user-level goodput over the Default vLLM-BoN baseline by 3.6% to 12.6% in our initial seed-0 experiments. At the same time, results on LongBench show that a single tail-oriented policy is not universal: prefill-heavy workloads can require a different recompute-aware policy. We therefore treat an **adaptive pressure selector** as a research extension rather than the main contribution. The main proposal is TailBlend; the adaptive selector is an experimental path toward generalizing beyond output-tail pressure.

## 2. Background and Motivation

### 2.1 Inference-Time Scaling and Best-of-N Sampling

Best-of-N sampling is one member of a broader family of inference-time scaling methods: rather than improving a model only by training a larger model, the system spends additional inference compute to generate, search over, verify, or aggregate multiple candidate outputs. Early evidence for this pattern appears in code generation and mathematical reasoning. The Codex evaluation introduced HumanEval and reported that repeated sampling can substantially increase the probability of finding a functionally correct program: the paper reports 28.8% HumanEval solve rate for one Codex sample and 70.2% with 100 samples per problem [9]. In math word problems, Cobbe et al. introduced GSM8K and used trained verifiers to select among multiple generated solutions, showing that generation plus verification can be more effective than relying on a single sample in a domain where candidate answers can be checked against labeled solutions [8]. Self-consistency later generalized this intuition for chain-of-thought prompting: it samples diverse reasoning paths and chooses the most consistent final answer, improving GSM8K by 17.9 percentage points in the reported experiments [7].

Recent work has made the same point more explicitly under the language of inference-time compute. Large Language Monkeys studies repeated sampling across tasks and models and reports that task coverage can scale over four orders of magnitude in the number of samples; in verifiable domains such as coding and formal proofs, coverage improvements can directly translate into task performance [3]. The paper also reports a SWE-bench Lite example where DeepSeek-Coder-V2-Instruct increases from 15.9% solved issues with one sample to 56% with 250 samples [3]. Snell et al. analyze test-time compute scaling for reasoning and compare mechanisms such as search against process reward models and adaptive response distributions; they find that the best use of test-time compute depends on prompt difficulty and report more than 4x efficiency improvement over a best-of-N baseline for math reasoning when compute is allocated adaptively [13]. These results motivate BoN as a practical serving workload: users may intentionally request many samples because those samples expose useful correctness coverage.

BoN is also important in alignment and preference optimization. Inference-aware fine-tuning studies a BoN setting where a verifier selects the best response from multiple LLM generations, and it fine-tunes models to optimize the inference-time BoN strategy itself [10]. BoNBoN Alignment analyzes the relationship between best-of-n sampling and RLHF-style alignment, showing that a best-of-n sampling distribution is closely related to an RLHF policy under a monotone reward transformation and deriving a fine-tuning method to mimic the best-of-n distribution without drawing `n` samples at inference [11]. Huang et al. give a complementary caution: increasing inference-time computation can help, but naive Best-of-N with an imperfect reward model can degrade because of reward hacking [12]. Together, these papers establish two points that matter for TailBlend. First, BoN is not an artificial benchmark trick; it is a widely studied way to convert inference compute into quality. Second, practical BoN serving systems must expect large and sometimes adaptive `N`, because the algorithmic value of BoN comes exactly from launching many candidate branches.

### 2.2 BoN as a Serving Workload: Many Branches, One User-Visible Parent

From the serving system's perspective, BoN changes the unit of work. A normal request contains one prompt and one sampled continuation. A BoN request contains one prompt and many continuations. The continuations are independent after they diverge, but they share a logical parent: the final user-visible result is produced only after the serving system has enough candidates for the verifier, reward model, voting rule, or pass/fail checker.

This parent-child structure is different from ordinary inter-request batching. In ordinary serving, each request has its own completion objective. In BoN, many child branches belong to the same user-level objective. If the serving system preempts a child branch that is the last unfinished branch of a parent, the parent may wait even though most of its branches have already completed. If it preempts an early branch from a parent with many remaining live branches, the user-visible effect may be smaller. This means that preemption should not be evaluated only by local branch age or local branch length. It should also consider the branch's marginal effect on the parent request.

The vLLM paper already identifies parallel sampling and beam search as advanced decoding algorithms with special memory-sharing structure: prompt KV can be shared, while autoregressive generation KV diverges across samples [1]. For BoN, this is exactly the operational shape. The prompt prefix is common, but each candidate's generated tokens create a separate growing KV footprint. As `N` increases, the logical parent becomes a group of live KV consumers rather than a single stream. TailBlend is built around this group structure: it keeps vLLM's normal eager execution model, but when memory pressure forces preemption, it scores child branches using both branch-level progress and parent-level completion signals.

### 2.3 Continuous Batching and the ORCA Baseline

Efficient LLM serving begins with the observation that autoregressive generation is iterative and variable length. ORCA showed that request-level batching is poorly matched to generative transformer serving because some requests finish earlier than others and newly arrived requests should not have to wait for an entire batch to finish [2]. ORCA proposed iteration-level scheduling, where the scheduler runs one model iteration at a time, and selective batching, where batching is applied only to a selected set of operations [2]. In its OSDI 2022 evaluation, ORCA reported large improvements over FasterTransformer on GPT-3 175B in both latency and throughput [2].

This line of work matters because TailBlend is not replacing continuous batching. It assumes the same serving world that ORCA helped establish: a live batch changes over time, requests have variable progress, and the scheduler makes repeated online decisions. TailBlend asks a narrower question inside that world. Given that a continuous-batching scheduler already has running branches and must make a memory-pressure decision, can it choose a better victim by using online BoN-specific state?

### 2.4 KV-Cache Management, PagedAttention, and vLLM Preemption

PagedAttention and vLLM identify KV-cache memory as a central bottleneck for high-throughput LLM serving. The vLLM paper notes that high throughput requires batching many requests, but KV cache is large, dynamically grows and shrinks, and can be badly wasted by fragmentation and redundant duplication [1]. For a 13B OPT model on an A100, the paper estimates that model weights occupy about 65% of GPU memory and dynamic request state, primarily KV cache, occupies close to 30% [1]. It also explains that output length is not known a priori, so a request's KV-cache demand grows during decoding and can exhaust physical blocks [1].

PagedAttention addresses this by dividing KV cache into fixed-size blocks and allowing those blocks to be stored in non-contiguous physical memory, analogous to virtual memory paging [1]. This design reduces internal and external fragmentation, allocates KV blocks on demand, and supports block-level sharing across sequences associated with the same request or across requests [1]. vLLM builds a serving engine around this memory manager and reports 2-4x throughput improvement over systems such as FasterTransformer and ORCA at similar latency, with larger gains for longer sequences, larger models, and more complex decoding algorithms [1].

However, PagedAttention does not remove memory as a hard constraint. The vLLM paper explicitly discusses scheduling and preemption: when request traffic exceeds capacity and vLLM runs out of physical blocks for new KV entries, it must decide which sequences to evict and how to recover evicted blocks [1]. vLLM adopts FCFS scheduling for fairness, serving earlier requests first and preempting later requests first [1]. It also uses all-or-nothing eviction at the sequence level because all blocks of a sequence are accessed together during generation [1]. Recovery can be done by swapping evicted KV blocks to CPU memory or by recomputing them later; the paper reports that recomputation can be efficient because the generated tokens can be concatenated with the original prompt and recomputed in one prompt-phase iteration [1].

This is the direct systems opening for TailBlend. vLLM already has a preemption point, and the original policy is primarily an online fairness heuristic. Under BoN, the "latest request" heuristic is not necessarily aligned with parent-level goodput because many requests are sibling branches of the same logical parent. TailBlend does not challenge PagedAttention; it uses the fact that vLLM has block-level preemption and asks whether the victim choice can be made more BoN-aware.

### 2.5 Prefill/Decode Interference and SLO-Oriented Serving

LLM inference has two phases with different performance profiles. In prefill, the model processes the prompt and creates KV state. In decode, it generates one token at a time and extends KV state. Sarathi-Serve and DistServe show that the interaction between these phases is a first-order serving issue, especially under latency SLOs.

Sarathi-Serve introduces chunked-prefills and stall-free schedules that add new requests to a batch without pausing ongoing decodes [14]. The key idea is to split a prefill into near equal-sized chunks and pair prefill chunks with decodes so large batch sizes can improve throughput while limiting the effect of batching on latency [14]. The OSDI 2024 paper reports higher serving capacity than vLLM under tail latency constraints across model and hardware settings [14]. DistServe takes a different approach: it disaggregates prefill and decoding onto different GPUs, because colocating both phases creates prefill-decode interference and couples resource allocation for the two phases [15]. DistServe optimizes placement and parallelism for TTFT and TPOT requirements and reports serving up to 7.4x more requests or 12.6x tighter SLO than state-of-the-art systems while staying within latency constraints for more than 90% of requests [15].

These systems provide an important boundary for TailBlend. They show that prefill pressure and decode pressure are distinct. TailBlend is mainly aimed at output-tail dominated BoN workloads, where sibling branches are already admitted and the dominant failure mode is decode KV growth and victim choice. Long-prompt workloads can behave differently: preempting a branch with a large materialized prompt or generated prefix can create recomputation cost that outweighs the benefit of evicting a predicted long tail. This is why the current proposal treats the adaptive prefill-aware selector as an extension rather than the main claim. The literature supports this caution: serving policies that work well for decode-dominated pressure are not automatically optimal when prefill interference or prompt recomputation dominates.

### 2.6 Prefix Sharing, Prompt Reuse, and Cache-Aware Scheduling

Several recent systems focus on reusing KV state across repeated or shared prefixes. This work is related to BoN because BoN branches share a prompt prefix, but the optimization target is different. SGLang introduces a frontend and runtime for structured language model programs; its runtime includes RadixAttention for KV-cache reuse and compressed finite state machines for structured output decoding [16]. In the NeurIPS 2024 paper, SGLang reports up to 6.4x higher throughput than state-of-the-art inference systems across tasks such as agent control, logical reasoning, few-shot learning, JSON decoding, RAG pipelines, and multi-turn chat [16].

Preble extends the prefix-reuse idea to distributed serving. It observes that modern prompts often include repeated instructions, tool examples, or long context, and proposes distributed prompt scheduling that co-optimizes KV-state reuse and computation load balancing [17]. Its evaluation reports 1.5x to 14.5x lower average latency and 2x to 10x lower p99 latency than state-of-the-art serving systems on real workloads and request arrival patterns [17]. HotPrefix studies prefix KV cache scheduling through a database-style cache-management lens. It identifies that high-hotness prefixes can be evicted prematurely, causing recomputation, and proposes dynamic hotness tracking, selective KV admission, and promotion for efficient prefix sharing [19].

TailBlend uses a different granularity. Prefix-cache systems decide which reusable prefix states should remain cached across requests or across program steps. TailBlend decides which live branch should lose its active decode KV when vLLM has already hit an allocation failure. Still, these papers provide a useful insight: KV state is not uniform memory. Some KV blocks are more valuable because they are likely to be reused, expensive to recompute, or critical to latency. TailBlend applies the same broad principle to BoN branches: a branch's KV state should be valued using progress, recomputation risk, and parent completion context rather than by request order alone.

### 2.7 Branch Parallelism and TAPER

The closest recent work to TailBlend's workload model is TAPER, which studies branch parallelism in LLM serving [4]. TAPER starts from the observation that recent methods expose intra-request parallelism in LLM outputs, allowing independent branches to decode concurrently. It argues that eager branch admission and conservative fixed caps are both brittle: eager admission can inflate shared decode-step latency and harm co-batched requests, while fixed caps can waste the throughput opportunity that motivated branch parallelism [4]. TAPER introduces a per-step admission controller that treats extra branches as opportunistic work and admits them only when the predicted branch externality fits within the current batch's slack budget [4]. On Qwen3-32B, the paper reports 1.77x goodput over IRP-Off and 1.48x over IRP-Eager while maintaining over 95% SLO attainment [4].

TailBlend is complementary rather than a replacement for TAPER. TAPER controls branch width before or during admission. Its key resource is time slack: if the batch has slack, more branches can be admitted; if not, branch width can be restricted. TailBlend deliberately does not regulate waiting admission and does not shrink decode width during normal execution. It intervenes only after vLLM has already failed to allocate KV slots. At that point, the question is no longer "how many branches should be admitted?" but "which already-running branch should lose its KV state?" This distinction is important because KV cache is accumulated work. Preempting a near-finished branch can waste computed state and delay parent completion, while keeping a predicted long-tail branch can continue to occupy memory for many more decode steps.

### 2.8 Output-Length Prediction, Tail Risk, and Online Sibling Evidence

Length prediction is a natural input to scheduling because output length affects runtime, KV growth, head-of-line blocking, and memory reservation. Recent work also shows why naive point prediction is fragile. Past-Future Scheduler argues that continuous batching depends on accurate estimates of request memory requirements; because output lengths vary, aggressive schedulers can cause harmful evictions while conservative schedulers can cause prolonged queuing, both hurting goodput under SLA guarantees [18]. Its ASPLOS 2025 paper estimates future memory occupancy from historical output-length distributions and reports 2-3x higher goodput than other schedulers under heavy load [18].

Robust Length Prediction argues that prompt-only length prediction should not treat one sampled length as a deterministic label. Even with a fixed model and decoding setup, the same prompt induces a prompt-conditioned output-length distribution, and the paper reports heavy-tailed behavior [5]. Scheduling LLM Inference with Uncertainty-Aware Output Length Predictions makes a similar scheduling argument: SJF-style policies can reduce head-of-line blocking, but a single point estimate does not match stochastic decoding, where output length depends on when EOS is sampled [6]. It models output length with a distribution and proposes Tail Inflated Expectation to account for long-output risk, reporting 2.31x lower per-token latency for online inference and 1.42x higher throughput for offline data generation in its evaluation [6].

TailBlend follows this distributional view but uses a BoN-specific online signal. Before any sibling finishes, TailBlend has little evidence and falls back toward the max-token budget. After some siblings finish, their observed lengths are samples from the same parent prompt under the same sampling configuration. They are not perfect predictions of the remaining siblings, but they are more local than a global prompt-level predictor. The shrinkage estimator in TailBlend is therefore conservative: it gradually trusts finished sibling lengths while retaining a prior when sibling evidence is sparse. This design is directly motivated by the length-prediction literature's warning that output length is uncertain and heavy-tailed.

### 2.9 Positioning TailBlend

The background literature leaves a narrow but important gap. ORCA and vLLM establish the serving substrate: iteration-level scheduling, paged KV memory, and preemption under memory pressure [1, 2]. Sarathi-Serve and DistServe show that prefill and decode phases can create different SLO bottlenecks [14, 15]. SGLang, Preble, and HotPrefix show that KV state should be managed with awareness of reuse and recomputation value [16, 17, 19]. TAPER shows that branch parallelism creates externalities and should be controlled under SLO constraints [4]. Length-prediction work shows that output tails matter and that point estimates can be misleading [5, 6, 18].

TailBlend occupies the missing preemption-specific point in this design space. It does not propose a new BoN algorithm, a verifier, a reward model, a prefix-cache system, or a branch-admission controller. It keeps Default vLLM-BoN semantics and changes only the victim selection rule at unavoidable KV allocation failure. The research hypothesis is therefore intentionally modest and testable: when vLLM must preempt a live BoN branch, sibling-finished lengths and parent-completion state can identify lower-harm victims than the default policy.

## 3. Methodology

### 3.1 Problem Setting

We study BoN serving inside vLLM. A user request has a logical parent id, represented in our implementation by `external_req_id`, and launches `N` child branches using vLLM's normal sampling API. We compare against **Default vLLM-BoN**, which is vLLM's ordinary `n=N` sampling path with the default scheduler and default preemption policy:

- vLLM eagerly starts BoN child branches when capacity allows.
- vLLM uses PagedAttention for KV-cache allocation.
- When `allocate_slots` fails, vLLM preempts a running request using its default victim policy.

In the default FCFS scheduling policy, this victim policy is order-based: when KV allocation fails, vLLM preempts the last request in the running queue. Under the priority scheduling policy, vLLM preempts the running request with the lowest effective scheduling priority, breaking ties by later arrival time. After preemption, vLLM frees the request's KV and encoder cache state, resets its computed-token progress, marks it as preempted, and places it back into the waiting queue for later recomputation. The default policy therefore does not use BoN-specific signals such as sibling completion, parent-level progress, predicted remaining output length, near-finish status, or recomputation harm when choosing the victim.

Our policies preserve the same eager BoN execution model. They only alter the victim selection rule when a preemption is already required.

### 3.2 TailBlend: Main Policy

TailBlend is a pressure-only preemption victim selector. It is called only after vLLM fails to allocate KV blocks for the current scheduling step. It does not defer waiting requests and does not shrink decode width on its own.

For each live branch `b`, TailBlend computes:

- `pred_remaining(b)`: predicted remaining output tokens based on online sibling-finished lengths.
- `remaining_norm(b)`: normalized predicted remaining output, defined as `pred_remaining(b) / max_tokens`.
- `output_progress(b)`: fraction of `max_tokens` already generated.
- `recompute_cost(b)`: fraction of prompt plus output budget already computed.
- `parent_rescue(parent(b))`: how close the logical BoN parent is to completion.
- `near_finish(b)`: whether the branch is likely to finish soon.
- `overdue(b)`: whether the branch is already beyond its SLO-derived deadline.
- `num_preemptions(b)`: how many times the branch has already been preempted.

The current TailBlend score is:

```python
score_tailblend =
    1.6 * remaining_norm
  - 1.2 * output_progress
  - 1.0 * parent_rescue
  - 0.8 * recompute_cost
  - 1.0 * near_finish
  - 0.4 * overdue
  - 0.2 * num_preemptions
```

Higher score means a better preemption victim. The score uses `remaining_norm` rather than raw `pred_remaining` so that requests with different `max_tokens` are comparable. The `near_finish` and `overdue` terms are binary indicators, while `num_preemptions` is a small anti-starvation penalty. Intuitively, TailBlend prefers to evict branches that are predicted to run long, have generated little useful prefix, are expensive for their parent to wait on, and do not appear close to releasing memory. It protects branches that are near completion, expensive to recompute, or likely to rescue the logical parent.

### 3.3 Online Tail Prediction from Finished Siblings

Exact output length is hard to predict before generation. TailBlend therefore uses an online estimator:

```python
if no sibling has finished:
    predicted_remaining = max_tokens - current_output_tokens
else:
    sibling_mean = mean(finished_sibling_lengths)
    weight = num_finished / (num_finished + shrink_k)
    prior = 0.5 * (current_output_tokens + max_tokens)
    predicted_final = (1 - weight) * prior + weight * sibling_mean
    predicted_remaining = max(0, predicted_final - current_output_tokens)
```

This estimator is intentionally conservative. Before there is sibling evidence, it assumes the branch may run to the max-token limit. As sibling completions accumulate, it gradually shifts toward the observed parent-specific length distribution.

### 3.4 Parent-Level Objective

A branch is not valuable only because it is short. In BoN, a branch is valuable when it helps complete the logical parent. TailBlend therefore includes `parent_rescue`, a simple parent-level completion signal:

```text
parent_rescue =
  (finished_siblings + near_finish_live_siblings)
  / (finished + live + queued siblings)
```

A high `parent_rescue` means the parent is already close to completion, so preempting one of its remaining useful branches may delay the user-visible result. TailBlend protects such branches.

### 3.5 Marginal-Harm View of TailBlend

TailBlend can be interpreted as an online approximation to a parent-level marginal harm objective. For a BoN parent `p` with branch set `B_p`, the user-visible completion time is governed by the slowest required branch:

```text
C_p = max_{b in B_p} C_b
```

Thus, a preemption decision should not be evaluated only by the local branch cost. The ideal victim under an SLO deadline `D_p` is the branch whose preemption causes the smallest loss in parent-level SLO feasibility:

```text
victim* = argmin_b Delta_b

Delta_b =
  Pr[C_parent(b) <= D_parent(b) | keep b]
  - Pr[C_parent(b) <= D_parent(b) | preempt b]
```

This quantity is not directly observable online because the system does not know future output lengths, future arrivals, or future KV pressure. TailBlend therefore constructs a victim-suitability surrogate `H(b)` from measurable branch and parent signals. Higher `H(b)` means lower expected parent harm and higher expected memory-turnover benefit:

```text
H(b) =
    lambda_1 * predicted_remaining(b)
  - lambda_2 * output_progress(b)
  - lambda_3 * parent_rescue(parent(b))
  - lambda_4 * recompute_cost(b)
  - lambda_5 * near_finish(b)
  - lambda_6 * overdue(b)
```

The branch with the largest `H(b)` is selected as the victim. The positive term identifies long-tail branches that are likely to keep KV memory occupied for a long time if retained. The negative terms protect branches whose preemption is expected to create large user-visible harm: branches with substantial computed state, branches likely to finish soon and release KV blocks, branches that may complete a nearly finished parent, and branches already under deadline pressure.

This objective yields a simple exchange argument. Consider two live branches `a` and `b` under the same KV allocation failure. Suppose `a` is near completion while `b` is a long-tail branch:

```text
predicted_remaining(a) << predicted_remaining(b)
```

If the default policy preempts `a` and keeps `b`, memory release is delayed until approximately `predicted_remaining(b)` more decode steps. If TailBlend preempts `b` and keeps `a`, memory can be released after approximately `predicted_remaining(a)` steps. Moreover, if `a` has already accumulated more computed tokens or belongs to an almost-complete parent, preempting `a` also incurs larger recompute waste or parent-level completion harm. Therefore, when:

```text
predicted_remaining(a) < predicted_remaining(b)
and (
  recompute_cost(a) >= recompute_cost(b)
  or parent_rescue(a) >= parent_rescue(b)
)
```

preempting `b` weakly dominates preempting `a` with respect to memory turnover and parent-level goodput. TailBlend performs this exchange locally at each unavoidable preemption point.

This does not claim global optimality. Instead, it provides a principled justification for the score: TailBlend is a pressure-only approximation to minimizing the marginal loss in BoN parent SLO feasibility. When there is no KV pressure, it behaves like Default vLLM-BoN because it does not change admission or normal decode scheduling. When pressure appears, it only changes the victim choice, limiting the blast radius of prediction errors to the branch that would have been preempted anyway.

### 3.6 Experimental Adaptive Extension

TailBlend is the main policy in this proposal. However, our initial results show that no single victim score handles every vLLM bottleneck. In particular, LongBench-style long-prompt workloads are prefill/recompute-heavy. For these cases, protecting already materialized prompt and generated prefix can be more important than evicting predicted long-tail branches.

We therefore implemented an exploratory adaptive selector. It computes two runtime pressure scores at each preemption event:

```text
prefill_pressure =
  kv_pressure * backlog_pressure * prompt_share * recompute_risk

output_tail_pressure =
  kv_pressure * output_share * tail_signal * tail_skew
```

The selector chooses:

- `prefill_aware` if prefill/recompute pressure is clearly dominant.
- TailBlend if output-tail pressure is clearly dominant.
- vLLM default preemption if the pressure type is ambiguous.

As an additional safety mechanism, we also tested a prefix-waste guard. The motivation is that even when the selector classifies an event as output-tail pressure, TailBlend can still choose a branch that has already accumulated a large generated prefix. In that case, preempting the branch may waste enough computed decode state to offset the expected memory-turnover benefit. The guard therefore blocks the TailBlend-selected victim and falls back to a more conservative choice if the victim's generated prefix is already large. This improved some mixed workloads but was threshold-sensitive. For this reason, the adaptive selector and prefix-waste guard remain research directions rather than part of the current main policy.

### 3.7 Current Implementation

The current implementation adds a TailBlend preemption module to vLLM. In the codebase, the feature is exposed as the experimental `tail_blend` scheduler policy.

- TailBlend: main output-tail-aware victim scoring.
- `prefill_aware`: recompute/prefix-aware victim scoring for prefill-heavy workloads.
- `adaptive`: experimental runtime selector between default, TailBlend, and prefill-aware.

Instrumentation records:

- total preemption count,
- predictor-selected preemption count,
- preempted computed tokens,
- preempted output tokens,
- adaptive selector counts for default/TailBlend/prefill decisions.

### 3.8 Preliminary Results

We evaluated Qwen/Qwen3.5-2B on four prompt sets with seed 0, request rate 1, and 40 prompts per dataset:

- GSM8K: math word problems.
- MBPP: Python programming prompts.
- LongBench/Qasper: long-context QA prompts.
- UltraChat: open-domain chat prompts used as a public chat workload.

These four datasets are not chosen as task-accuracy benchmarks. They are chosen to stress different vLLM serving regimes. GSM8K and MBPP represent verifiable reasoning and coding workloads where BoN often creates long output tails. LongBench/Qasper represents long-prefill pressure, where recomputing prompt and prefix state can dominate the benefit of output-tail victim selection. UltraChat represents a mixed conversational workload with more heterogeneous prompt and response shapes. This split helps test whether the policy is genuinely scheduler-aware rather than overfit to one math-style workload.

For GSM8K, MBPP, and UltraChat, we use `max_output_tokens=4096` and `max_model_len=5120`. For LongBench/Qasper, we use `max_output_tokens=2048` and `max_model_len=8192`.

#### Main policy: Default vLLM-BoN vs TailBlend

| Dataset | N | Default vLLM-BoN Goodput | TailBlend Goodput | Delta | Default vLLM-BoN SLO | TailBlend SLO |
|---|---:|---:|---:|---:|---:|---:|
| GSM8K | 8 | 2802.41 | 3037.46 | +8.4% | 82.5% | 92.5% |
| GSM8K | 16 | 1941.17 | 2186.62 | +12.6% | 45.0% | 55.0% |
| MBPP | 8 | 2212.23 | 2303.87 | +4.1% | 72.5% | 72.5% |
| MBPP | 16 | 1494.05 | 1547.23 | +3.6% | 50.0% | 50.0% |
| LongBench/Qasper | 8 | 1453.52 | 1428.53 | -1.7% | 87.5% | 90.0% |
| LongBench/Qasper | 16 | 534.37 | 469.49 | -12.1% | 30.0% | 25.0% |
| UltraChat | 8 | 2006.91 | 1887.32 | -6.0% | 72.5% | 72.5% |
| UltraChat | 16 | 1218.74 | 1256.24 | +3.1% | 40.0% | 45.0% |

These results support the main hypothesis for output-tail workloads. TailBlend improves GSM8K and MBPP consistently and improves UltraChat at `N=16`. At the same time, LongBench demonstrates a limitation: when pressure is dominated by long prompts and recompute risk, a tail-oriented victim score can be worse than vLLM's default policy.

#### Exploratory extension: Adaptive selector with prefix-waste guard

| Dataset | N | Default vLLM-BoN Goodput | TailBlend Goodput | TailBlend-Adaptive Goodput | Adaptive vs Default vLLM-BoN | Adaptive vs TailBlend |
|---|---:|---:|---:|---:|---:|---:|
| GSM8K | 8 | 2802.41 | 3037.46 | 3111.65 | +11.0% | +2.4% |
| GSM8K | 16 | 1941.17 | 2186.62 | 1975.78 | +1.8% | -9.6% |
| MBPP | 8 | 2212.23 | 2303.87 | 2326.46 | +5.2% | +1.0% |
| MBPP | 16 | 1494.05 | 1547.23 | 1558.34 | +4.3% | +0.7% |
| LongBench/Qasper | 8 | 1453.52 | 1428.53 | 1408.61 | -3.1% | -1.4% |
| LongBench/Qasper | 16 | 534.37 | 469.49 | 531.88 | -0.5% | +13.3% |
| UltraChat | 8 | 2006.91 | 1887.32 | 2001.99 | -0.2% | +6.1% |
| UltraChat | 16 | 1218.74 | 1256.24 | 1179.86 | -3.2% | -6.1% |

The adaptive selector is promising but not stable enough to be the main contribution. It recovers the major TailBlend failure on LongBench `N=16`, but threshold-sensitive guard behavior hurts GSM8K `N=16` and UltraChat `N=16`. We therefore position adaptive scheduling as a research proposal: the pressure taxonomy appears useful, but the selector requires principled calibration or online learning.

## 4. Research Questions

This proposal focuses on the following questions:

1. Can online sibling-length signals improve vLLM BoN preemption victim selection without changing eager BoN admission?
2. Under which workload regimes does TailBlend improve user-level goodput over Default vLLM-BoN?
3. Which preemption metrics best explain success or failure: preemption count, preempted computed tokens, preempted output tokens, or parent completion rate?
4. Can an adaptive pressure classifier select between TailBlend, recompute-aware, and default preemption without dataset-specific thresholds?

## 5. Project Timeline

The report deadline is September 5, 2026. The schedule below reserves the final two and a half weeks for writing, figure cleanup, and reruns, rather than treating experiments as finished only at the deadline.

### June 1 - June 9: Stabilize the Main Implementation

- Finish renaming and cleanup around the `tail_blend` scheduler policy.
- Keep TailBlend separate from experimental adaptive logic in both code and reporting.
- Add focused unit tests for parent grouping, finished-sibling length tracking, remaining-length estimation, and victim-score ordering.
- Verify that TailBlend changes only the preemption victim after `allocate_slots` failure and does not change normal eager BoN admission.

### June 10 - June 20: Instrumentation and Reproducibility

- Add or validate counters for total preemptions, predictor-selected preemptions, preempted computed tokens, preempted output tokens, parent completion time, and parent SLO attainment.
- Standardize benchmark scripts, output directories, random seeds, model configuration, and dataset sampling.
- Produce a small smoke-test suite that can be run before every large experiment.
- Freeze the first reproducible experimental configuration by June 20.

### June 21 - July 5: Main Repeated-Seed Evaluation

- Run Default vLLM-BoN and TailBlend on GSM8K, MBPP, LongBench/Qasper, and UltraChat.
- Repeat at least three seeds for each dataset and `N in {8, 16}`.
- Track mean, variance, and confidence intervals for user-level goodput, SLO attainment, preemption count, and recompute waste.
- Decide whether the 7-term TailBlend score should remain the main policy or be simplified into a smaller three-signal score based on ablation evidence.

### July 6 - July 17: Ablations and Sensitivity Analysis

- Compare TailBlend against ablations: remaining-only, remaining+recompute, remaining+parent, and full TailBlend.
- Sweep `N in {4, 8, 16, 32}` on the strongest output-tail workloads.
- Sweep request rate and GPU memory utilization to identify the pressure range where TailBlend helps most.
- Analyze whether improvements come from fewer preemptions, less recompute waste, faster parent completion, or better SLO rescue.

### July 18 - July 31: Failure-Mode and Generalization Study

- Focus on LongBench/Qasper and other long-prompt cases to quantify prefill/recompute-heavy failures.
- Separate prompt/prefix recomputation cost from output-tail KV pressure using runtime counters.
- Evaluate at least one additional model size if GPU time allows.
- Produce failure-mode plots that explain when TailBlend is expected to lose to default or prefill-aware victim selection.

### August 1 - August 9: Adaptive Selector Decision Point

- Calibrate the adaptive pressure selector using only training-style development runs, not final held-out summaries.
- Test leave-one-dataset-out behavior across GSM8K, MBPP, LongBench/Qasper, and UltraChat.
- Decide by August 9 whether adaptive is strong enough to be a secondary contribution.
- If adaptive is unstable, report it as exploratory evidence and keep TailBlend as the only main policy.

### August 10 - August 16: Final Experiment Freeze

- Rerun the final selected experiment matrix with fixed scripts and fixed seeds.
- Regenerate all tables from raw JSON/CSV outputs.
- Check consistency between benchmark logs, reported metrics, and paper tables.
- Freeze final results by August 16 unless a correctness bug is discovered.

### August 17 - August 27: Report Drafting

- Write the full report narrative: motivation, related work, methodology, experimental setup, results, ablations, and limitations.
- Convert raw results into final tables and figures.
- Emphasize the main claim: pressure-only BoN-aware victim selection improves output-tail workloads without changing eager admission.
- Clearly separate main TailBlend results from adaptive-selector exploration.

### August 28 - September 3: Revision and Polish

- Tighten terminology around BoN parent/branch, preemption, recomputation, prefix, and prefill.
- Verify every citation and remove claims not directly supported by results or cited papers.
- Add limitations and threats to validity, including seed count, model size, workload selection, and coefficient sensitivity.
- Ask for external feedback and revise the report for clarity.

### September 4 - September 5: Final Submission

- Perform final proofreading and formatting.
- Recheck all table values against the saved experiment outputs.
- Submit the report by September 5, 2026.

## 6. Expected Contributions

1. A vLLM-native analysis of BoN serving bottlenecks under KV-cache pressure.
2. TailBlend, a pressure-only predictor-aware preemption victim selector for BoN serving.
3. An online sibling-length estimator that uses completed BoN branches without requiring a trained prompt-level length predictor.
4. A parent-level completion-aware preemption objective for user-level goodput.
5. A preliminary pressure taxonomy for future adaptive vLLM BoN scheduling.

## 7. References

[1] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, Ion Stoica. "Efficient Memory Management for Large Language Model Serving with PagedAttention." SOSP 2023. https://arxiv.org/abs/2309.06180

[2] Gyeong-In Yu, Joo Seong Jeong, Geon-Woo Kim, Soojeong Kim, Byung-Gon Chun. "Orca: A Distributed Serving System for Transformer-Based Generative Models." OSDI 2022. https://www.usenix.org/conference/osdi22/presentation/yu

[3] Bradley Brown, Jordan Juravsky, Ryan Ehrlich, Ronald Clark, Quoc V. Le, Christopher Re, Azalia Mirhoseini. "Large Language Monkeys: Scaling Inference Compute with Repeated Sampling." arXiv:2407.21787, 2024. https://arxiv.org/abs/2407.21787

[4] Swapnil Gandhi, Siva Hari, William J. Dally, Christos Kozyrakis. "Regulating Branch Parallelism in LLM Serving." arXiv:2605.06914, 2026. https://arxiv.org/abs/2605.06914

[5] Jing Wang, Yu-Yang Qian, Ke Xue, Chao Qian, Peng Zhao, Zhi-Hua Zhou. "Robust Length Prediction: A Perspective from Heavy-Tailed Prompt-Conditioned Distributions." arXiv:2604.07931, 2026. https://arxiv.org/abs/2604.07931

[6] Haoyu Zheng, Yongqiang Zhang, Fangcheng Fu, Xiaokai Zhou, Hao Luo, Hongchao Zhu, Yuanyuan Zhu, Hao Wang, Xiao Yan, Jiawei Jiang. "Scheduling LLM Inference with Uncertainty-Aware Output Length Predictions." arXiv:2604.00499, 2026. https://arxiv.org/abs/2604.00499

[7] Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V. Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, Denny Zhou. "Self-Consistency Improves Chain of Thought Reasoning in Language Models." ICLR 2023. https://arxiv.org/abs/2203.11171

[8] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, John Schulman. "Training Verifiers to Solve Math Word Problems." arXiv:2110.14168, 2021. https://arxiv.org/abs/2110.14168

[9] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. "Evaluating Large Language Models Trained on Code." arXiv:2107.03374, 2021. https://arxiv.org/abs/2107.03374

[10] Yinlam Chow, Guy Tennenholtz, Izzeddin Gur, Vincent Zhuang, Bo Dai, Aviral Kumar, Rishabh Agarwal, Sridhar Thiagarajan, Craig Boutilier, Aleksandra Faust. "Inference-Aware Fine-Tuning for Best-of-N Sampling in Large Language Models." ICLR 2025. https://proceedings.iclr.cc/paper_files/paper/2025/hash/c40bed606c51c8e827c1ba75aa2da054-Abstract-Conference.html

[11] Lin Gui, Cristina Garbacea, Victor Veitch. "BoNBoN Alignment for Large Language Models and the Sweetness of Best-of-n Sampling." NeurIPS 2024. https://proceedings.neurips.cc/paper_files/paper/2024/hash/056521a35eacd9d2127b66a7d3c499c5-Abstract-Conference.html

[12] Audrey Huang, Adam Block, Qinghua Liu, Nan Jiang, Akshay Krishnamurthy, Dylan J. Foster. "Is Best-of-N the Best of Them? Coverage, Scaling, and Optimality in Inference-Time Alignment." ICML 2025. https://proceedings.mlr.press/v267/huang25c.html

[13] Charlie Snell, Jaehoon Lee, Kelvin Xu, Aviral Kumar. "Scaling LLM Test-Time Compute Optimally Can be More Effective than Scaling Parameters for Reasoning." ICLR 2025. https://proceedings.iclr.cc/paper_files/paper/2025/hash/1b623663fd9b874366f3ce019fdfdd44-Abstract-Conference.html

[14] Amey Agrawal, Nitin Kedia, Ashish Panwar, Jayashree Mohan, Nipun Kwatra, Bhargav Gulavani, Alexey Tumanov, Ramachandran Ramjee. "Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve." OSDI 2024. https://www.usenix.org/conference/osdi24/presentation/agrawal

[15] Yinmin Zhong, Shengyu Liu, Junda Chen, Jianbo Hu, Yibo Zhu, Xuanzhe Liu, Xin Jin, Hao Zhang. "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving." OSDI 2024. https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin

[16] Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue Sun, Jeff Huang, Cody Hao Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E. Gonzalez, Clark Barrett, Ying Sheng. "SGLang: Efficient Execution of Structured Language Model Programs." NeurIPS 2024. https://proceedings.neurips.cc/paper_files/paper/2024/hash/724be4472168f31ba1c9ac630f15dec8-Abstract-Conference.html

[17] Vikranth Srivatsa, Zijian He, Reyna Abhyankar, Dongming Li, Yiying Zhang. "Preble: Efficient Distributed Prompt Scheduling for LLM Serving." arXiv:2407.00023, 2024. https://arxiv.org/abs/2407.00023

[18] Ruihao Gong, Shihao Bai, Siyu Wu, Yunqian Fan, Zaijun Wang, Xiuhong Li, Hailong Yang, Xianglong Liu. "Past-Future Scheduler for LLM Serving under SLA Guarantees." ASPLOS 2025. https://arxiv.org/abs/2507.10150

[19] Yuhang Li, Rong Gu, Chengying Huan, Zhibin Wang, Renjie Yao, Chen Tian, Guihai Chen. "HotPrefix: Hotness-Aware KV Cache Scheduling for Efficient Prefix Sharing in LLM Inference Systems." Proc. ACM Manag. Data 3, 4 (SIGMOD), Article 250, 2025. https://doi.org/10.1145/3749168
