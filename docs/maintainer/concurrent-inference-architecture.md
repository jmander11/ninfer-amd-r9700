# NInfer Small-Scale Concurrent Inference Architecture

This document defines the execution architecture by which NInfer supports a small number of concurrent requests on a single GPU with a single model instance. A typical `max_concurrency` is 2–16.

The design goal is not to let multiple requests execute in turn, but to let all requests in the decode phase form one true batched model execution: one model traversal, one Device Graph replay, and one set of batched operators produce results for multiple requests at the same time.

This document defines only decisions that affect module boundaries, scheduling semantics, resource ownership, or execution correctness. Protocol error formats, allocator implementation, kernel organization, target-specific profile values, and test plans belong to the corresponding serving, runtime, operator, and qualification documents.

---

## 1. Scope

### 1.1 Supported workload

- Single GPU, single resident model instance;
- `max_concurrency=C` is fixed at startup; ordinary and MTP support `C=1..16`, and DFlash supports `C=1..12` (`kDFlashMaximumConcurrency`; speculative option validation rejects a larger `C` at startup); typical `C=2..16`;
- `0..C` resident requests at run time; RAM suspension keeps the admitted identity, and the total number of logical requests is still bounded by `C + max_pending_requests`;
- Text and image/video prompts;
- ordinary decoding and engine-wide speculative decoding;
- streaming and non-streaming output;
- prefix reuse, plus an optional pinned-host RAM tier for completed bundles and live suspension;
- Different prompt lengths, context lengths, generation limits, and sampling configurations.

### 1.2 Non-goals

- General priority/preemptive scheduling, pausing at arbitrary execution points, and disk swap of active requests; only the settled-boundary RAM suspension of §5.8 is supported;
- Multi-request batched prefill (there is only one prefill owner at a time; the mixed round of §8.9 merges only one bounded slice of that owner into the target forward of a DFlash decode round);
- Multi-GPU or distributed inference;
- Priority, tenant QoS, or deadline-aware GPU scheduling;
- General continuous batching for tens to hundreds of requests;
- Dynamically capturing Device Graphs for new shapes during serving.

### 1.3 Required sequence-state substrate

The concurrent engine requires the lower layer to provide a concurrent-capable sequence-state substrate, and it assumes that this contract is already satisfied. The substrate must provide:

- Shared, block-addressable growing-context storage that does not depend on a slot or on a per-request physically contiguous range;
- Fixed-size recurrent/backend state allocated per sequence;
- Unified reserve, commit, truncate, and release semantics for growing and fixed state;
- Pin, ownership-transfer, and eviction semantics for retained state;
- An opaque per-sequence state view that attention and other model operators can consume directly;
- State handles and a logical valid frontier that are stable during one GPU execution unit.

This document defines how the concurrency layer reserves, holds, batches, and transfers this state. It does not define the substrate's page geometry, physical slab layout, allocator, block-table representation, or page-aware operator implementation. The active contract of the growing KV substrate is defined by [Paged KV Context Store](paged-kv-cache.md); this document consumes only its allocation, reservation, and execution-view semantics.

---

## 2. Core invariants

### 2.1 Maximal batched decode

At any round boundary, if there are `B>0` resident, funded, non-transfer-held decode-ready requests, the next decode round must include all `B` requests. Growth-blocked requests first obtain a window per §5.8. Splitting runnable rows into multiple request-local forwards is forbidden, and so is executing them one after another in order to reuse the single-request path.

### 2.2 Boundary-only membership

A request joins or leaves active execution only at a boundary after a GPU round or prefill chunk completes. While GPU work is in flight, slot binding, batch membership, and sequence-state ownership do not change.

### 2.3 Fixed slots, shared state memory

`max_concurrency` fixes the number of control slots and does not divide KV/context memory evenly among slots. KV cache, GDN state, and speculative-backend state are provided by the Sequence-State Store of §1.3. A slot holds only an allocation handle and a reservation, and does not own a static context partition.

The Engine configuration must be able to hold `C` active fixed-size sequence states, `C` rows of execution metadata, and one shared executor workspace at the same time; otherwise that `max_concurrency` is invalid. Retained state stays only in the fixed backing of a currently idle lane and occupies its actually mapped growing state; a new admission can claim or evict it, so it cannot lower the guarantee for `C` active sequences. Growing context is still shared capacity, so there is no promise that any `C` long contexts can be admitted at the same time.

### 2.4 Admission preserves output limits and a completion path

With RAM off, admission commits GPU capacity for the prompt, the complete effective output bound, and transient backend growth. With RAM on, the complete prompt and the current output window obtain GPU entitlement; dynamic RAM credits and the completion cohort preserve the completion path of §5.8. Window exhaustion produces no terminal output and does not shorten the request's output/context budget. A continuation is paused only at a settled boundary, without discarding or replaying already generated output.

### 2.5 One prefill owner

At most one admitted request owns the prefill/finalization path at a time. Suffix prefill advances in bounded steps: with `mixed_forward=0`, the owner's entire prefill completes contiguously before the next decode round (prefill-first); with `mixed_forward=N>0` and a decode-ready request present, the forward width of each step is N: a separate step advances at most N prompt tokens, and a mixed step advances N minus the round's live verify columns (§8.9); steps interleave with decode rounds or are merged into a mixed round per §7.3. Other waiting requests stay in the host queue and occupy no slot or model state.

### 2.6 Single GPU execution owner

One GPU executor serially submits all prefill chunks and decode rounds, and is the only modifier of active slots, sequence state, and the next batch. CPU preparation, request ingress, and output I/O may run in parallel, but they cannot directly advance model state.

The D2H/H2D transfers of the RAM second tier go through an independent copy engine (`DeviceContext::copy_stream`). They are not a GPU scheduling unit and do not occupy the compute owner. A copy may overlap with a legal compute unit of **another** lane; pages that the current PrefillChunk / DecodeRound is reading or writing must not be a copy source or destination. The MTP/DFlash prefill context-checkpoint freeze is the exception: the D2H of the staging GDN (slot `C`, disjoint from the lane's current), the staging hidden, and the DFlash staging cyclic state may overlap with the next prefill chunk of the **same** lane. During the freeze, `C` is borrowed from the turn-rollback occupant, and the H2D after packing restores the rollback; `occupied=false` must happen before the clobber D2D. Rewrite-checkpoint capture likewise borrows the staging: the compute stream first waits for the staging's previous copy (a stream wait that does not block the host), a D2D copies the lane's current GDN and the DFlash local lane into the staging, and `copy_stream` then copies the staging with layer-by-layer linear D2H into that lane's pinned rewrite image, overlapping with subsequent prefill/decode. `synchronize()` on `device.stream` drains only compute, not copy_stream. Checkpoint host images (each lane's rewrite image and the MTP/DFlash context-checkpoint head pool) are carved at startup from one driver-backed HIP/GTT pinned slab and are owned by the Program; a head returns to the pool after it leaves the queue, and the subsequent freeze, turn-rollback, and RAM/disk restore reuse the same image and completion event, so `hipHostMalloc`/`hipHostFree` are not called during serving. The pool capacity is one rollback head per lane, plus the number of ladder heads that all lanes can hold at the same time within the shared KV capacity (marks above `--max-context` are unreachable); when the pool is exhausted, an optional capture or restored head is skipped. Waiting for the previous DMA/host-copy owner before reusing a head is always a stream wait (copy_stream or the compute stream waits on the head fence); the scheduler thread does not host-wait.

### 2.7 Bounded ingress and output

Both the Engine and the HTTP server limit the number of generation request lifetimes to `max_concurrency + max_pending_requests`. Each request also has a finite input size, a finite effective output token bound, and an absolute pending deadline; media preparation is serialized by a single permit. Sustained ingress can therefore only fill a finite set of request records and cannot build an unbounded wait queue.

The total upper bound of owning results comes from "finite number of requests × finite output per request"; result/event storage belongs to the respective request and continues to count toward request lifetime capacity until the response consumer releases it.

### 2.8 Per-request semantic isolation

Sampling, RNG, stop conditions, generation limit, usage, and output state belong to the request. They must not depend on which slot or compact batch row the request currently occupies, and they must not affect other requests in the same batch.

### 2.9 Protected, resource-aware admission

Prepared request order is the fairness baseline for admission. RAM off uses the full-lifetime protected-head backfill of §5.2–§5.7; RAM on uses the finite window-pressure epoch and oldest-suspended resume precedence of §5.8. Neither may let sustained fresh ingress postpone the progress of existing requests indefinitely.

RAM-off backfill changes only the admission order of waiting requests and does not take resources that an active request has committed but not yet materialized. RAM-on suspension may release a victim's GPU entitlement, but must first secure the complete live continuation. All funded, resident, decode-ready requests still enter the same maximal compact batch; growth-blocked and transfer-held requests do not enter membership.

### 2.10 Constrained generation and bounded recovery

The opt-in request reasoning-token budget is owned by the output session, separately from
framing-inclusive reasoning usage. Committed reasoning-body IDs consume the cumulative allowance;
rejected speculative proposals, prompt history and framing do not. Hypothesis masks carry the
same accounting and accepted prefixes end at the policy boundary. The next target position permits
only the canonical reasoning close; that token passes through target commit and KV state normally.
Pending UTF-8 and partial textual framing are finalized in the reasoning channel. Policy-forced
framing has intervention counters rather than a fabricated model-confidence logprob. Ordinary
caller stops, cancellation and completion/context ceilings retain precedence.

The private clean-retry comparison uses immutable executor startup selection, one budget-triggered
retry and the existing shared retry/output ceilings. It retains the original sampling profile and
replaces the capped attempt with a labeled engine notice plus a native non-thinking continuation.
It withholds replacement answer/tool deltas until terminal validation, and treats any exhausted
replacement as failure. With ordinary generation recovery disabled, this does not activate replay,
repetition or duplicate-tool recovery and a failed replacement exhausts instead of retrying again.
The output-session ledger stays on the common request through KV suspension, and is copied into
replacement sessions; neither restore nor prompt replay resets it. No mechanism has corpus or live
qualification yet; compare them in the authorized test window before selecting deployment behavior.

For a request with a reasoning cap in the ordinary Close configuration, automatic clean retries
are disabled. Confirmed reasoning repetition keeps its analysis and continues toward natural or
capped close; cycle escape and framing repair remain active. Invalid replay continuations,
persistent answer repetition and proved duplicate-tool cycles fail explicitly rather than discard
the attempt and retry. The private Retry comparator remains separate and bounded.

The Qwen frontend owns compiled tool schemas, transactional grammar state, and typed completed calls. Program owns planner-accounted device eligibility masks and pinned exchange storage. Ordinary sampling uses one mask per compact row; speculative verification uses one per actual chain position or tree node. A Program-owned matcher thread walks the proposal tree depth-first on the committed grammar state, accepting each node's token and rolling it back, once proposal metadata reaches the host mailbox, while the target forward runs; target sampling acquires the masks. Proposals do not advance committed grammar; only the accepted publication transaction does. Matcher errors surface before publication.

P-less sampling evaluates the eligible full-vocabulary softmax. With temperature T>0, collision mass L=sum(p²), and epsilon=0.0625, membership is p>=max(L*exp(-2*epsilon/T),1/1024). The floor is not relaxed and there is no top-20 support cap. Temperature and request seed remain active; top-k, top-p, min-p and presence/frequency penalties do not. Temperature zero remains eligible argmax with lower-token-id ties. A thinking-only cycle detector can exclude the predicted continuation of the least repeated suffix period 32..2048 (Hamming allowance floor(2p/512)). Exclusion removes an atom after the support decision, without recomputing L; an emptied support falls back to the eligible mode or its runner-up. Only the first position of a speculative round uses this cycle-exit restriction.

For non-raw, text-only thinking requests, bounded recovery may withhold a repeated completed call or interrupt persistent reasoning at a committed round boundary, independently of sampler and temperature. Reasoning detection requires three disjoint exact occurrences of a 256-token passage and at least 4096 distinct redundant tokens. Both reasoning and answer detectors consume channel-attributed accepted tokens before publication, and latch their proof across reasoning-close and terminal boundaries. A reasoning proof withholds the current round's unpublished answer/tool deltas before retry or error. Only duplicate-tool recovery inserts rejected-call feedback; a reasoning retry never fabricates tool feedback from an unexecuted call in that round. Without a supported retry context, proved reasoning repetition fails explicitly. Raw output is excluded; free-text answer monitoring also excludes schema/tool output. This is not a reasoning-length limit. The detector resets per attempt. The family owns the original input. A retry splices onto the resident prompt tokens, keeping historical reasoning: it closes the previous assistant prologue with its matching terminator, appends a labeled system notice or rejected-call feedback, and opens the selected native generation prologue. Only the failed generation is omitted; a second retry appends its notice after the first retry's prompt. It invents no user turn or call result. A splice whose suffix does not round-trip through the tokenizer exhausts recovery.

With generation recovery enabled, a thinking response can quarantine its initial answer while its generated token IDs exactly copy the preceding generated reasoning passage (at least 256 tokens). Genuine reasoning still streams normally. A divergent answer, an extension beyond the passage, or an uncorroborated terminal copy releases held text and its original logprobs unchanged. Only a complete exact copy licenses one additional dedicated `</think>` token: that consumed token confirms a framing replay, discards the held copy, and rearms quarantine. The extra close is not published or fed to ordinary tool grammar, but remains in generated IDs, KV state and completion usage. Another complete copy, or natural termination without a separate answer or selected tool after repair, requests an unpublished retry with a labeled engine notice. Caller stops and cancellation retain precedence. Repair and proof boundaries end the accepted speculative prefix; hypothesis masks license the extra close only at the matching boundary. Raw, schema-constrained and tool-argument contents are excluded. The original output budget and shared two-retry cap still apply; exhaustion is explicit. Intentionally requesting a long exact reasoning copy with an extra close can trigger this policy. After a confirmed framing replay or persistent reasoning loop triggers a retry, the new assistant continuation uses the model's native non-thinking prologue. This retry strategy remains active for later retries on the same request; it does not itself establish framing-replay evidence. The original input and historical reasoning stay intact; only the new generation mode changes. The frontend carries the already-corroborated immutable reasoning reference into these replacements, so a complete copy of that same passage triggers recovery without licensing another close. A subsequent closed-thinking prompt is ended with a turn terminator alone, preserving all earlier notices without inserting an extra reasoning close. All replacement answer/tool deltas and their logprobs remain withheld until ordinary terminal repetition/tool validation accepts the attempt; reasoning remains visible. A failed replacement is discarded and may retry only within the remaining shared budget. Explicit caller stops release held text; cancellation releases text while omitting provisional tool events. For an initially non-greedy p-less request, these non-thinking recovery attempts use standard sampling at temperature 1.0, top-k 20, top-p 0.95, min-p 0 and zero presence/frequency penalties, retaining the resolved request seed. Initial generation and standard/greedy request profiles are unchanged. The actual retry profile and `enable_thinking` mode are recorded on `retry_started` recovery events. A replacement ending naturally without non-whitespace prose in answer mode or a complete selected tool call requests a bounded retry with an `empty_replacement` cause; terminal framing alone is not answer text. Exhausting the output/context budget without either fails explicitly; caller stops and cancellation retain precedence. This is repetition recovery and publication validation, not a check that an answer or external task is correct.

The target-owned output session filters publication after the runtime attaches accepted-token logprobs. It retains complete deltas while quarantined, preserving their order and metadata on release. The frontend owns frozen reasoning IDs and transactional replay cursors; Program transports their per-hypothesis masks through the existing exchange, including plain chat. Grammar state skips only the corroborating repair close. Runtime owns retry admission, accounting and the shared retry budget. Preview rollback restores replay state, and commit ends at repair or repeated-copy proof before a novel continuation can share discarded deltas.

At most two retries consume the reserved retry service work and the original remaining output budget, each within the original context entitlement. A retry keeps its lane/admission and leaves the decode-ready set. With prefix reuse enabled it retains the lane in place and plans the spliced prompt against it: a ready prompt-frontier checkpoint (the live frontier, the response-replay rewrite checkpoint, or a context checkpoint) is restored and only the suffix is prefilled, with the typed FP8-K/INT4-V pages, GDN state, and MTP/DFlash proposal state of that checkpoint. Only when no resident checkpoint is reusable does it take the longer RAM or disk image onto the aborted lane, and a failed host restore cold-prefills the same lane without sticking a cold fallback. It never trims KV at an arbitrary token while retaining stale recurrent state, and it captures no turn rollback checkpoint. Other requests keep maximal-batch scheduling outside that exclusive prefill; a host-restore retry holds its lane in copy-hold like an admission, so they keep decoding while the RAM or disk image copies. Calls remain unpublished until accepted; already streamed prose/reasoning cannot be retracted. Extra prefill work and recovery events are reported separately from original prompt usage. Exhaustion is a request error, not synthetic EOS, engine shutdown, or a promise of progress.

---

## 3. Overall architecture

```text
 clients / CLI / OpenAI / Anthropic
                  │
                  ▼
┌──────────────── Server Frontend ─────────────────────┐
│ validate · bounded CPU preparation · ordered pending │
│ queue · protected-head resource backfill             │
│ finite count/per-request bytes/deadline · responses  │
└───────────────────┬──────────────────────────────────┘
                    │ admission at a boundary
                    ▼
┌──────────────────── GPU Executor ────────────────────┐
│ Boundary Coordinator · Scheduler · Slot Table[C]     │
│                         │                            │
│          ┌──────────────┴──────────────┐             │
│          ▼                             ▼             │
│  PrefillChunk(one sequence)   RoundMembership[B]    │
│          │                             │             │
│          │                    Target Batch Assembler │
│          │                             │             │
│          └───────────┬─────────────────┘             │
│                      ▼                               │
│       shared Model Runtime + DecodeBatchFrame[C]     │
│       weights · workspace · graph families           │
│                      │                               │
│                      ▼                               │
│               per-row round results                  │
└───────────────┬───────────────────┬──────────────────┘
                │                   │ committed output
                ▼                   ▼
┌─ Sequence-State Store prerequisite ─┐  Server Frontend
│ Text KV                            │  / async output
│ GDN / recurrent state             │
│ speculative-backend state         │
│ state entitlements                │
│ optional retained prefixes        │
└──────────────────────────────────────┘
```

The responsibilities of the components are as follows:

| Component | Responsibility |
|---|---|
| Server Frontend | Request validation, bounded CPU preparation/pending work, cancellation input, and response I/O |
| GPU Executor | Admission, boundary processing, state commits, and all GPU submission |
| Slot Table | Holds the stable control state of admitted requests |
| Scheduler | Performs protected-head admission at a boundary, and selects the next `PrefillChunk`, `MixedRound`, or the complete active `DecodeRound` |
| Target Batch Assembler | Converts round membership into the typed controls and state selectors the target needs |
| Model Runtime | Holds the single resident model, shared execution memory, and graph assets, and executes the whole-batch schedule |
| DecodeBatchFrame | One address-stable round staging area with maximum capacity `C`; each round uses only the exact-`B` prefix |
| Sequence-State Store | Prerequisite storage substrate; owns per-sequence model state and capacity |

These are responsibility and ownership boundaries; they do not require C++ classes with the same names.

All product entrypoints submit requests to the same GPU Executor; a caller or connection thread does not drive the model loop directly, nor does it own a Model Runtime of its own. The `C` concurrent requests share the same weights, workspace, round frame, and graph families, rather than constructing `C` single-request Programs.

Blocking `Engine::generate` may be kept as a product facade, but its semantics are to submit an owning request record and wait for that request to complete, not to run the complete generation loop inside the caller thread. One Engine-owned GPU worker executes the global boundary loop; concurrent callers wait on their own result/output queues. The GPU worker only appends committed output events and does not call network writes or a potentially blocking `OutputSink`.

---

## 4. Request and slot model

### 4.1 Request lifecycle

```text
RECEIVED
   │ validation and bounded pending capacity
   ▼
WAITING
   │ bounded CPU preparation, then boundary admission:
   │ slot/lane + state entitlement
   ▼
PREFILL
   │ zero or more suffix chunks, then one finalization unit
   │ exact retained hit skips suffix work, not finalization
   ├── terminal ──────────────────────────────────► MODEL_FINISHED
   ▼
DECODE_READY
   │ participates in successive batched rounds
   ├────────────────────────────────────────────► DECODE_READY
   │ EOS / stop / limit
   ▼
MODEL_FINISHED
   │ response may continue draining
   ▼
RESPONSE_FINISHED
```

Cancellation or failure can terminate a request from any phase. `WAITING` includes both bounded CPU preparation and waiting for GPU admission; these are server-side phases and do not need to be expanded into more model-execution states.

`DECODE_READY` is a stable state: the request already owns the complete committed model frontier and the single current token/anchor needed by the next decode round, and its sampling, RNG, and published output are consistent with that frontier. Whether it belongs to the current in-flight round is recorded by that round and is no longer modeled as a long-lived request state.

RAM on additionally permits `DECODE_READY -> GROWTH_BLOCKED -> CAPTURING -> SUSPENDED -> RESTORING -> DECODE_READY`; a funded extension returns `GROWTH_BLOCKED` directly to `DECODE_READY`. Capture/restore holds retain the original admitted request and its output owner. Suspension releases the physical lane only after capture completes; resume may bind a different lane and does not repeat prefill/finalization or select the pending token again. Admitted suspended time is not pre-admission queue time and does not restart a queue deadline.

Intermediate prefill chunks only advance prompt state. The finalization unit performs target output selection and establishes the decode anchor; if a terminal condition is hit immediately, the request does not enter a decode batch. When the retained current frontier exactly covers the complete prompt, suffix prefill may be skipped, but the next anchor is still produced from the retained tail state through the same finalization unit, and under MTP the exact-hit bridge/proposal is completed. On a hit on a saved rewrite checkpoint, the new suffix is prefilled from that checkpoint. A KV match alone is not sufficient to enter `DECODE_READY`.

### 4.2 Slot

A slot is a physical control position leased to one resident request. The logical request identity, output, RNG, and budget persist across suspension; resume may rebind the request to a different slot. A resident slot stores or references:

- Request identity and lifecycle state;
- Sampling, RNG, stop, and generation-limit state;
- Per-request output session, usage, timing, and speculative statistics;
- A handle to the `SequenceState` in the shared pool;
- Prompt/decode progress required by the scheduler;
- Cancellation and terminal flags.

A slot does not own:

- A fixed proportion of context capacity;
- A permanent batch row;
- Shared model workspace;
- Response buffers after model completion;
- Logical ownership of the retained `SequenceState` after model completion.

After a request reaches `MODEL_FINISHED`, the model result is handed to the response path, and the slot can be reused at the current or a later boundary. Network completion is not part of the slot lifetime. The slot index is also not an external request identity; the cancellation or completion of an old request cannot act on a later occupant.

The current runtime maps a control slot one-to-one to a physical execution lane, but the slot, the external request identity, and the compact batch row remain three distinct identities. Fixed-size device state can be reserved per lane and rebuilt when the lane is occupied by a new request. After a request completes, the active slot is released immediately; if a prefix is retained, the `SequenceState` can remain in the backing of that idle lane, but no active request control exists any more and it does not count toward active concurrency.

### 4.3 SequenceState

`SequenceState` is one continuable model state defined by the target. An occupied slot has sole write authority over it, and it contains or references at least:

- Main Text and optional MTP KV allocations and their committed frontiers, plus fixed DFlash cyclic state when that backend is selected;
- Linear Attention and other fixed model-state allocations;
- The target decode cursor, including the current anchor and position/RoPE progress;
- Hidden/checkpoint state needed for continuation;
- Prefix identity and target-defined reusable checkpoints.

`SequenceState` does not contain stop/output/transport state, and does not own a batch row, round activations, logits, shared workspace, or a graph. The current fixed-state backing is lane-affine: ordinary consuming reuse stays on the same lane, and the planner searches all free lanes for a reusable continuation and, once one is selected, establishes the new request control on the same lane; when active capacity is needed, retained state on other free lanes may be evicted first. The sampling, RNG, stop, and output state of a new request are always recreated. Disposable forks copy into an independently owned destination lane under the source-preserving transaction described in §6.4.

For Qwen3, the lane is the sole locator of Linear Attention state. With `C=max_concurrency`, the shared pool always uses `[0,C)` as the current committed state of each lane; an MTP or DFlash engine additionally reserves slot `C` as the Engine-wide GDN: the default hot occupant is turn-rollback (the append occupy pins current+`tail_hidden` at the previously completed `E` before suffix prefill), which is borrowed by the ladder freeze and then reinstalled; a rewrite capture borrows it and keeps it as the device copy of that lane's rewrite image (tagged by lane and image generation), from which a rewrite restore can D2D directly. One slot selects the convolution history and recurrent state of all GDN layers at once. A rewrite checkpoint itself is not a slot: each lane owns, from startup, one pinned host image (the GDN slot image, plus the cyclic local lane image under DFlash) and one copy event; when the staging no longer holds that generation, restore waits for that event on the compute stream and then performs a layer-by-layer H2D. A decode round does not maintain, in `SequenceState`, a state selector that varies with the speculative position.

The Program initializes the text and speculative backend pools' physical KV payload planes and every DFlash local K/V lane's complete padded payload once at startup, even when Device Graph capture is disabled. Tier images and disposable forks copy whole pages and complete fixed images; unused page tails and cyclic positions therefore contain defined bytes without extending valid frontiers or the live attention window. Block tables and unrelated arena storage are outside this initialization. The Program also initializes each complete ordinary, MTP, and DFlash decode egress frame once, so full-frame host transfers include defined inactive entries and padding.

### 4.4 Batch row

A batch row exists only for one decode round. At each boundary, the Target Batch Assembler re-establishes a compact mapping:

```text
batch row -> slot -> sequence-state handle
```

For example:

```text
slots:          [0] [1] [2] [3]
decode-ready:    Y   N   Y   Y

logical rows:   [0] [1] [2]
row -> slot:     0   2   3
logical B:       3
```

An empty slot does not enter the model batch, so slot reuse or holes do not require a different execution model.

---

## 5. Request ingress and admission

### 5.1 Bounded ingress

Before expensive preparation, the Server Frontend validates request syntax, model capability, input size, and the finite generation bound. Only requests that pass validation can obtain bounded CPU preparation and pending capacity.

The host side sets finite limits on the following resources:

- the number of generation-request lifetimes, fixed at `C + max_pending_requests`;
- body/media bytes and media item count per request;
- CPU preparation work that holds request records at the same time;
- one serial media-input preparation permit;
- the longest time from a request obtaining lifetime capacity to its admission;
- the effective output token bound per request.

When the request lifetime count is full, a new generation request is rejected immediately as overload. There is no additional unbounded overflow queue; the per-request input limits together with the finite outstanding count bound host input memory, instead of a separate aggregate byte allocator.

CPU preparation produces an owning, immutable prompt representation. Prepared requests enter the Engine ordered admission queue in preparation-completion order, so a request waiting for the media permit does not hold the first position of the prepared queue. The queue timeout starts when a request first obtains request lifetime capacity, so it includes media acquisition, prompt preparation, and Engine admission waiting.

This order is a fairness baseline, not an absolute head-of-line barrier. Only the protected-head backfill defined in §5.3–§5.6 can let a later request be admitted first; CPU preparation completion, prefix reuse, or the request content itself does not create any other implicit priority.

### 5.2 Authoritative admission feasibility

Sections 5.2–5.7 define **RAM-off** full-lifetime admission and its protected-head proof. Bounded ingress in §5.1 applies in both modes. RAM-on admission, growth, suspension and fairness are defined in §5.8; its current window is never a completion reservation or a donor-completion claim.

The Scheduler does not re-derive whether a request can enter from token counts or GPU memory bytes. The target/resource layer gives each prepared request an authoritative entitlement vector, already rounded to the physical granularity of each pool. Logically it contains at least:

```text
E(request) = {
    active lane:                  1,
    Main Text page groups:       E_main,
    optional MTP page groups:      E_mtp,
    other request-time units:    target-defined, if any
}
```

Each dimension may be compared only with capacity of the same kind; Main/MTP pages cannot be summed by bytes or borrowed from one another, and tail-page slack of an allocation is not usable capacity. The current fixed recurrent/backend backing is established for `C` copies and transfers ownership with the physical lane, so it is expressed by the `active lane` dimension; the GPU memory bytes of fixed backing are not counted again in the dynamic entitlement. Graph, workspace, round frame, and weights are engine-fixed resources; owning result capacity is already obtained at ingress and is likewise not part of this vector.

A candidate has current admission feasibility only if all of the following hold:

1. a control lane is available;
2. the request is valid for the fixed model, frontend, and selected execution backend;
3. after active-priority retained eviction, every entitlement dimension can commit the request's full-lifetime requirement;
4. there is currently no other prefill owner and no unfinished copy-hold admission; admission immediately establishes the request's suffix-prefill or exact-hit finalization ownership, or first enters copy-hold until the relevant RAM copies permit `start_prefill_lane`.

MTP adds its typed growing-page entitlement; DFlash fixed cyclic state is already part of the startup-frozen lane backing. The Scheduler does not establish different queue policies for ordinary, MTP, or DFlash. Even though startup sizing guarantees that the MTP pool does not become normal backpressure before the Main pool, admission still consumes the full authoritative vector, and an early MTP failure is a sizing/accounting invariant violation.

Admission is one atomic boundary transaction: the prepared entry, the optional retained-state claim, the slot/lane, and all fixed/growing state entitlements are either obtained together and published as an admitted request, or the request-visible ownership is left unchanged. The feasibility probe and protected-head shadow accounting allocate no pages, pin no retained entries, and give a queued request no partial ownership; the same authoritative transaction still re-checks and commits at the end.

A request must first pass exclusive feasibility: with all evictable retained state evicted and no other active request, it can occupy the Engine's admission exclusively. A request that cannot be held exclusively is rejected permanently and cannot become a protected head; if there is no active request and the queue head still cannot enter, the cause can only be a permanent rejection or a resource-accounting failure, and the Engine must not wait idle.

### 5.3 Ordered queue and protected head

Each admission opportunity first checks the oldest prepared entry:

```text
oldest head is feasible now
    -> admit head

oldest head is permanently infeasible / cancelled / expired
    -> remove it, then reconsider the new oldest entry

oldest head is exclusively feasible but blocked by active lane/entitlement
    -> protect this head and consider qualified backfill
```

If the head's full lane/entitlement is already feasible and only the existing one-prefill-owner rule prevents admission, it remains an ordinary waiting head, establishes no protection epoch, and does not scan later candidates; at the next legal admission turn after the prefill owner clears, the head is retried first. Protection is established only when there is no prefill owner and the head genuinely needs an existing active request to release a lane or entitlement, so the frozen donor set is non-empty.

Only the oldest head is protected at any one time. Protection is bound to a stable request generation identity, not to a reclaimable lane, slot index, or compact batch row. It is only a Scheduler shadow state and does not pre-claim real pages, a lane, or a retained continuation for the waiting request.

The protection lifecycle is:

```text
UNPROTECTED
    │ oldest head is transiently blocked at an admission boundary
    ▼
PROTECTED_OPEN
    │ protected resource opportunity has matured,
    │ but head is still blocked by a temporal borrower/prefill owner
    ▼
PROTECTED_DRAIN
```

`PROTECTED_OPEN` allows the backfill defined in §5.5. `PROTECTED_DRAIN` forbids all later/backfill admission, advances only the already admitted bounded prefill/decode work, and keeps first-retrying the protected head at every legal admission turn. Successful admission of the head, cancellation, queue timeout, or permanent rejection clears the current protection; the next oldest request can then establish its own epoch, but does not inherit the old frontier, work credit, or shadow ledger. Multiple waiting large requests do not establish nested reservations.

The queue deadline is not reset by protection or by being bypassed. Backfill only guarantees that, under normal Engine progress, continuing later ingress does not cause unbounded starvation; it does not promise that the protected request is admitted before its configured queue timeout.

### 5.4 Frozen release frontier and persistent-safe backfill

When protection is established, let:

- `K` be the per-dimension usable capacity;
- `H` be the protected head;
- `A` be the frozen incumbent identities already admitted at this moment;
- `D` be the minimal ordered prefix of `A` that is expected to complete earliest and whose release is sufficient for `H` to be admitted.

The Scheduler orders `A` by the remaining-service projection of §5.5 and adds donors one at a time until the following relation holds in every dimension:

```text
sum(E(a), a in A without D) + E(H) <= K
```

`D` defines a set of incumbent completion/release events, not an externally promised wall-clock ETA. Once established it can only shrink because of early incumbent completion/cancellation; later admitted backfill can never become a donor, and cannot swap the frontier for a later active request. If the early release of any other incumbent already makes `H` feasible, the Scheduler may admit `H` earlier than the frozen frontier.

Let `P` be the backfill requests that are still active and classified as persistent-safe. They must always satisfy the cumulative shadow invariant:

```text
sum(E(a), a in surviving members of A without D)
+ sum(E(p), p in P)
+ E(H)
<= K
```

A candidate can become a persistent-safe backfill only if it both `fits-now` and still satisfies this per-dimension relation after being added. All still-surviving `P` must be accounted cumulatively; multiple candidates cannot each be compared against the same initial surplus. When a persistent-safe request completes it is removed from the ledger, and its surplus can be used by another qualified request; but at any resource-release boundary that is also an admission turn, `H` must be retried first before the surplus is considered for reuse; otherwise only the release/maturity is recorded, and `H` is retried first at the next legal turn.

This invariant guarantees that even if all `P` are still active when the donor frontier arrives, `H` does not lose the lane, Main pages, or optional MTP pages it needs because of them. It does not guarantee that `H`'s wall time is unaffected; the bounded prefill of backfill and the larger decode batch still change incumbent round latency.

In protection accounting, retained state is an evictable cache, not queued ownership. `E(H)` uses the full entitlement of a cold admission; no matching lane/checkpoint is pinned for `H`. Retained state produced after an incumbent or backfill completes must, when an active admission needs it, be evicted at the same boundary under active-priority semantics, so its full active entitlement can be regarded as releasable. Non-evictable engine-fixed occupancy belongs to the baseline outside `K` and cannot be disguised as a donor release.

### 5.5 Service projection and bounded temporal borrowing

Total entitlement determines resource feasibility but cannot by itself represent residence time. The Scheduler uses a finite, monotonic service projection expressed in uniform positive-integer scheduling-work quanta to identify relatively short work:

```text
service work = known Text/Vision suffix-prefill and finalization work
             + effective remaining output reservation expressed as decode work
```

The output part uses the declared finite effective output bound; it does not predict prompt content, reasoning difficulty, or the actual EOS. The prefill part uses the already bounded target scheduling-unit profiles; a short output cannot offset arbitrarily long Text/Vision prefill. The current Qwen3 profile uses the finite upper bound of externally scheduled prefill/finalization steps as the prefill quanta and counts each effective remaining output token as one decode quantum; the prefill quanta simulate step by step, under the same chunk policy used at execution (including the irregular-tail 4096 rule), the splits caused by rewrite checkpoints, Vision items/MTP shifted consumers, and the historical assistant turn-closure frontiers of multimodal prompts, rather than adding only one quantum per split. The selected speculative backend can influence the target's uniform work projection, but does not create MTP/DFlash policy branches in the Scheduler.

The projection serves only the following two purposes: choosing the frozen incumbent donor order, and constraining temporal borrowing. It is not a completion ETA, a memory-safety basis, or a fairness proof. An active request can reach EOS early on the next round, and speculative acceptance and batch round latency also vary; estimation errors must be absorbed by the protection state rather than by assuming the prediction is accurate.

Let `W_frontier` be the projected remaining service distance for the frozen donors, advancing in parallel, to reach the required release event when the epoch is established. Protection initializes a positive, finite, non-replenished temporal credit with it:

```text
credit_initial = W_frontier
credit_after_admission = credit_before_admission - service_work(candidate)
```

The service work of every admissible request is at least one quantum; even an exact retained hit with no suffix tokens still has finalization work. The credit also uses the same discrete unit, so each temporal admission deducts at least one unit, and there is no sequence that decreases forever without being exhausted. A candidate that cannot satisfy the persistent-safe invariant but whose current full entitlement `fits-now` may temporarily borrow the critical capacity that `H` needs after the frontier only if all of the following hold:

1. the candidate's full service work does not exceed the current projected remaining distance, and it is expected to complete before the frozen frontier;
2. that service work does not exceed the remaining temporal credit;
3. after admission the candidate's full work is deducted from the credit, and it is not refunded if the request completes early;
4. temporal admission stops immediately once the protected opportunity has matured.

Such a request is called a temporal backfill; its full active footprint still enters the current authoritative resource accounting, but not the persistent-safe ledger, because the policy expects it to release before the frontier. The work credit prevents a 32K long request and a 2K short request from being treated as the same single bypass, and also limits the cumulative service work that later requests can borrow within one epoch. Credit only decreases and never increases; a new arrival cannot extend the epoch.

The Scheduler scans the bounded pending snapshot in prepared order and selects the oldest qualified candidate. Qualification may skip an earlier entry that is unsafe or not short enough, but does not permanently apply SJF, best-fit, or knapsack to the whole queue. If all requests declare the same huge default output bound, the Scheduler has no reliable evidence for judging actual difficulty; temporal qualification is then naturally conservative, and provable persistent-safe backfill can still be used; it must not guess from prompt content.

### 5.6 Admission boundary and progress rules

A boundary is an admission turn only if it satisfies the following condition:

```text
no prefill owner
and no copy-hold admission
and (no decode-ready request or the completed GPU unit was a DecodeRound)
```

A copy-hold, like a prefill owner, holds the admission/prefill opportunity exclusively: it is not a GPU unit, but a second request must not be admitted before the D2H/H2D copies complete and `start_prefill_lane` is called. A held lane is neither decode-ready nor a prefill owner.

Other boundaries may process completion, cancellation, timeout, and protection bookkeeping, but cannot commit head or backfill admission. This gate guarantees that existing decode-ready donors complete at least one progress round between two admissions; another request cannot be admitted immediately and consecutively after a final prefill/finalization ends.

One admission turn successfully admits at most one request. Cleanup of cancellation, timeout, and permanent-invalid entries does not count as admission; the Scheduler may continue looking for a new head/candidate in the same frozen pending snapshot, but a candidate that fails under the same resource state is not retried repeatedly at that boundary.

The order of each admission opportunity is fixed as:

```text
1. resolve completed GPU unit
2. finish/cancel active requests and release-or-retain their state
3. remove visible pending cancellation, timeout and permanent failures
4. if this is an admission turn, retry the protected head or the oldest FIFO head
5. on the same turn, if an open protected head remains blocked,
   admit at most one qualified backfill
6. choose and launch one next GPU unit
```

If exact current accounting already makes the protected head feasible, it is always admitted before any later request. If the frozen donors have already released, or the current non-temporal shadow resources are already sufficient to hold `H`, but `H` is still blocked by a temporal borrower or the existing prefill owner, protection immediately enters `PROTECTED_DRAIN`. Drain does not preempt the borrower; it forbids later/backfill admission, but the protected head is still retried first at every legal turn. A persistent-safe borrower that is still in prefill may also briefly block head admission, but one-prefill-owner work is finite and no other request can be admitted during that time.

Completion/cancellation of a frontier member counts as a release only after its state is actually released at a boundary. If that boundary is an admission turn, the head is retried first immediately; otherwise maturity is updated, and the head is retried first at the next legal turn. An active request's used/reserved transition does not change its entitlement; a future reservation that has not yet been materialized is not tail surplus. Backfill completion retention also cannot override head-first active-priority eviction.

Protection does not change the GPU progress policy:

- between two successful backfill admissions, every surviving frozen donor takes part in at least one maximal `DecodeRound`;
- each successful decode/speculative round of every non-terminal row commits at least one output token, otherwise it is directly terminal;
- each `PrefillChunk` advances a positive, finite amount of prompt work, or completes finalization;
- each request has a finite output bound, and each GPU unit has a bounded execution extent.

Therefore frozen donors release within a finite number of rounds. The temporal admissions that can happen before the frontier are also finite; a miss after the frontier matures enters drain, and the already admitted borrowers release resources after eventual completion/cancellation. Sustained ingress cannot roll the donor frontier backward, and cannot form a resource deadlock. This guarantee is eventual progress, not zero wall-time interference or an admission ETA.

### 5.7 Waiting outcomes and sustained ingress

Admission outcomes are classified as:

| Condition | Outcome |
|---|---|
| request is invalid or exceeds a semantic limit | permanent rejection |
| cannot be held exclusively even after active-priority eviction | rejected for the current Engine configuration |
| request lifetime capacity is full | ingress rejects as overload |
| can be held exclusively, but the current slot/resource/prefill owner does not allow admission | protected waiting or ordinary waiting |
| a qualified later request uses the currently available backfill opportunity | admitted before the blocked head |
| queue timeout expires before admission | ends with queue-timeout |
| Engine is stopping or unavailable | rejected as unavailable |

Only an admitted request gets a GPU completion commitment. A queued request has only a bounded waiting commitment, with no execution commitment or admission ETA. It waits until the first of: boundary admission, cancellation/disconnect, queue timeout, or Engine shutdown/failure.

Under sustained ingress, once the finite queue is full, subsequent requests are rejected immediately. A later arrival can backfill within an open protection epoch through the same persistent-safe/temporal qualification, but cannot change the protected identity, increase the temporal credit, become a current frontier donor, or increase the workload/resource commitment of any admitted request.

---

### 5.8 RAM-on window pressure and live suspension

Admission selects prefix reuse before funding its live-image credit. Credit allocation can evict unclaimed optional prefixes even when funding fails. If the RAM index changes, reselect the source and fund its actual image at the same output window before claiming it or shrinking the window. A shorter prefix can carry more checkpoint heads and require more backing.

Nonzero `--kv-ram-capacity` automatically enables output windows of at most 8192 tokens; there is no separate selector. Initial admission funds the complete prepared prompt and a bounded output window. Target-owned `ExecutionWindow` values specify exact rounded Main/backend entitlements, cumulative funded output and suspension bytes. Effective output/context limits remain lifetime bounds. Every next legal unit must fit its funded physical writes, including DFlash verification width and MTP provisional lead even when fewer output tokens may commit. A smaller useful window may be granted under pressure.

At settled boundaries, resolve output/cancellation and transfer readiness, then retry growth. One growth-blocked lane waits while executable peers advance. A queued replacement may take a blocked lane only after an atomic plan proves useful post-transition execution and secured victim storage. Unblocked lanes are not suspended solely for ingress. Two blocked lanes or no executable work opens a pressure plan: reclaim optional retained state first, then choose the smallest feasible victim set, preferring blocked victims, lower transfer bytes and less repeated suspension. Completion-cohort members are excluded until an atomic funded handoff makes them ordinary backed residents. Planned released capacity is held for its selected growth/replacement transaction until completion or rollback.

At the first blocked or suspended request, freeze admitted incumbent identities and their oldest identity. At most `C-1` subsequent fresh admissions may bypass that epoch; early completion does not refund credit. Two blocked lanes, no executable work or exhausted credit enters sticky drain and stops fresh replacements. The epoch resets only when its oldest incumbent completes/cancels, never on fresh arrival, physical resume or a newer completion. Resume the oldest suspended request before fresh ingress, except an already selected replacement whose victim is being captured. A resumed victim advances a useful output chunk before another capture unless no executable work remains. Existing transfers are awaited while unrelated compute continues; they are not a license for capture/restore cycling.

Every resident outside the completion cohort holds allocatable RAM credit through its funded window end; suspended identities retain complete live images. Credits include physical packing/alignment, required checkpoint/recovery state, speculative tails and transfer-held allocations. Optional completed/system-prefix entries yield to live ownership. Fragmented free-byte totals alone cannot establish a credit. The sum of the cohort members' target-produced completion resource envelopes fits the shared GPU capacity in every pool and physical-lane axis. This is a future feasibility proof, not a current reservation: all requests acquire only their current windows. Backed peers can evacuate into secured RAM, allowing the cohort to grow and finish together without another RAM allocation. Oldest feasible residents join the cohort and release unnecessary credits. A member can leave for suspension only after securing its image credit and atomically exchanging its completion claim with a feasible progressing successor. A restoring request claims future cohort capacity before transfer starts and retains its image backing until restore finishes; completion, cancellation and rollback release claims exactly once.

A completion envelope includes native recovery suffix growth only when the actual retry policy permits it. Raw requests, absent recovery context, disabled general recovery without private budget-Retry, and public capped Close use their ordinary lifetime vectors. Otherwise the Qwen frontend bounds native suffix growth across both permitted attempts, framing phases, notices and the proof context's canonical known duplicate-call feedback; the target adds the original output allowance once and applies its context/resource limits. Every actual retry's lifetime vector must fit its immutable admitted envelope. Exceeding the envelope is an invariant failure; it never grants extra capacity or silently changes the retry/output contract.

Before parking a live request for queued admission or pressured growth, the worker can reclaim unused future window entitlement. The target plans and commits each shrink while retaining all mapped current/speculative pages and at least one useful next output token. The common scheduler uses those exact resource vectors to choose bounded donors and preserve the largest window end that satisfies the concrete transaction, then resizes backed donors' credits. Lifetime output/context limits and completion claims do not shrink. A released window is spent only by the selected admission/growth transaction; the worker does not preempt an unblocked request merely to fill ingress.

Capture occurs after committed decode resolution and deferred GDN folds, never from provisional output or in-flight prefill. It preserves exact current three-plane KV/publication state, recurrent state, hidden/anchor, backend state, rope/ledger identity and required checkpoints. Original Request/OutputSession, RNG, grammar/stop/tool buffering, budget, recovery and published output remain owned by the same request. Copies fence source lifetime; a failure retains the resident source or suspended image. Live images are not prefix lookup entries, optional evictions or disk-spill candidates. No cold re-prefill fallback is permitted for active continuation. Cancellation/shutdown releases every image, credit and lane exactly once.

This is the implemented contract under qualification. GPU correctness, live-transfer failure cases and aggregate throughput measurement remain deferred until the user authorizes tests; no measured throughput improvement is claimed.

## 6. Shared context and state memory

This section defines only the resource contract and ownership transitions by which the concurrent engine uses the Sequence-State Store; the storage substrate itself is provided in advance by §1.3.

### 6.1 State ownership

Each admitted request obtains one sequence-state ownership on the selected free lane. Depending on the selected model mode, that state contains:

- the Text KV cache;
- GDN convolution/recurrent state;
- speculative-backend persistent state;
- position and model-continuation metadata.

Growing KV pages come from shared pools and are not statically partitioned per lane, so different requests can occupy very different amounts of context memory. Fixed recurrent/backend backing is bound to the current lane; this is a physical placement and does not bind an external request or a compact row identity to a lane.

### 6.2 Completion reservation

This full-lifetime reservation section applies with RAM off. With RAM on, the same physical pool accounting applies to the currently granted window; §5.8 defines dynamic growth and its progress guarantee. A window ending is a blocked/growth transition, while only the lifetime output/context limit is terminal.

Admission computes a target-specific entitlement for each request. The logical upper bound for Main Text is:

```text
prompt_tokens + effective_output_tokens - 1
```

It is rounded up to the page size. The DFlash Full backend uses the same entitlement; the MTP backend also covers up to `K-1` provisional leading positions, and is truncated by the per-sequence `max_context`. Prefix reuse changes only the already-materialized portion and the prefill work; it does not lower the entitlement that a new request can ultimately reach.

These target facts, rounded by each pool's page geometry, form the authoritative admission vector of §5.2. The same vector is used for the actual reservation, the protected frontier, and the persistent-safe shadow ledger; the Scheduler does not maintain a second token or byte estimation formula.

Every class of shared state resource must always satisfy:

```text
active used capacity
+ active reserved-but-not-yet-used capacity
+ retained used capacity
<= total usable capacity
```

The optional MTP KV pool is physically separate from Main Text but is sized by the same target-specific profile. Let `S=max_context`, `P` be the page size, `L=ceil(S/P)`, `C=max_concurrency`, `M_min=max(L,C)`, and `M_max=C*L`. The Explicit policy derives `M=ceil(K_main/P)` from the user's token capacity; the Automatic policy reserves headroom `R` after weights are loaded and solves directly, from the reservation curve of the complete target physical layout, for the largest `M` that fits in `F-R`. CLI/server default to `R=64 MiB` (`--kv-capacity-headroom`) and report the actual free memory after the full startup. Main Text uses `M` physical page groups and each allocation has logical capacity `L`; MTP uses `M + C*ceil((K_draft-1)/P)` physical groups, with the same logical capacity `L`, where `K_draft` is the speculative draft window. The extra groups hold only the provisional lead of multiple concurrent MTP rows and do not enlarge the request context. DFlash2 has no growing page pool; its BF16 cyclic state is fixed at startup. Startup requires `K_main>=S`, `M>=C`, and `M<=C*L`; `K_main>=S` applies only to Explicit. After capacity resolution, all typed pools and the Graph topology are fixed and are not expanded at request time. After active-priority retained eviction, any active entitlement set that satisfies the advertised Main pool contract must also satisfy the backend reservation; an earlier backend failure is a startup sizing or accounting invariant violation, not normal admission backpressure.

Prefill/decode progress only converts the logical reservation the request already holds into used capacity and creates no new resource requirement. Whether physical blocks are set aside in advance or bound on demand belongs to the Sequence-State Store.

Prefix reuse atomically adjusts the retained entitlement on the same lane and converts it to an active entitlement, without double charging. When a request ends, a retained entry keeps only the pages that are already mapped and does not keep future growth entitlement; eviction releases the whole continuation before active admission is retried.

Before starting a prefill chunk or a decode round, the Scheduler uses the request's remaining reservation and its logical context/output limit to bound the maximum possible state growth of that unit. A request with no legal room to grow becomes terminal directly and must not start a unit that could overrun.

This decouples concurrency from context length:

- one request can use most of the pool's capacity;
- two requests can use completely different amounts of capacity;
- when the capacity left by active reservations is insufficient, a new request waits;
- an active request never discovers late in its run that its declared output cannot be accommodated.

Therefore every request must have a finite maximum output bound. When the caller does not provide one, the Engine uses the configured generation cap.

### 6.3 Fixed and growing state

Growing state such as the KV cache is charged by request context. Fixed-size recurrent/backend state is allocated per admitted sequence. Even if the two come from different internal pools, both must be satisfied in the same admission decision.

Growing KV state is carried by the [Paged KV Context Store](paged-kv-cache.md) in target-defined typed page-group pools. A sequence always holds a Main Text allocation; MTP Engine additionally owns an MTP allocation, while DFlash owns only its separate fixed cyclic state. Each growing pool has an independent frontier and reservation; the physical context of a single request need not be contiguous. The concurrent engine operates only on the bundle handle, the logical frontiers, and the reservation vector, and does not participate in page or physical address management.

While GPU work is in flight, the meaning of the per-request state handle and view must stay stable. The contracts for page size, slab layout, block table, and allocator belong to the Paged KV Context Store and are not repeated in this concurrency document.

### 6.4 Prefix reuse

A retained prefix is a single-owner SequenceState detached from a finished request. It stays on its original physical lane, but that lane's control slot is free to the scheduler. Retained state publishes only checkpoints for which the target has saved the complete continuation state. The current Qwen3 retained state can publish the current resume frontier, one typed rewrite checkpoint when valid, and, under MTP or DFlash, the prefill context-checkpoint ladder heads on the live lane. A rewrite is marked `TurnClosure` or `ResponseReplay` according to its purpose at capture time; the two kinds are mutually exclusive and reuse the same physical payload, and additionally store the corresponding recurrent, hidden, speculative-backend, and position state, without copying the KV payload. A ladder head freezes the current GDN at the end of the committed prefill chunk and the hidden of that chunk's last column; the KV remains a prefix of the same bundle.

The Qwen frontend publishes the desired checkpoint from the effective `preserve_thinking` semantics: `false` selects the end of the first assistant opener after the last real user turn; `true` selects the end of the complete deterministic generation prologue of this request, i.e. the current prompt frontier. Thinking generation includes `<think>\n`, and non-thinking generation includes the complete empty thinking block. Exception: when the latest assistant turn of a thinking request has no reasoning (discarded by the client), the next-turn re-render diverges right after the opener (the effort template omits the empty wrapper; the `<think>\n\n` of the toggle template is tokenized separately), so the checkpoint is placed at the generation opener (before `<think>\n`), at the cost of one extra very short prefill unit. The boundary is first produced as a byte offset, then independently tokenized and verified to be a complete prompt token prefix; the schema adapter does not infer or rewrite these target-private semantics. With `false`, the frontend additionally publishes the turn-closure token frontier at the end of each historical assistant opener (tokenized in turn order over disjoint ranges and verified against the complete encoding). Cold prefill of a multimodal prompt splits chunks at these frontiers so that its state matches the checkpoints captured in earlier turns; a suffix of a multimodal prompt that contains no Vision item still executes with the prompt's 3-axis MRoPE positions, not 1-D text RoPE. A Vision chunk is split at the first image Text column to match a reused text-only prefix, while that chunk still carries the image embedding required by the MTP shifted input.

When admission succeeds on the same lane, it consumes the retained entry and transfers SequenceState ownership to the new request:

- when the incoming prompt ends at the current resume frontier, suffix token processing is skipped, and the request enters `DECODE_READY` after a finalization unit produces the next anchor;
- when the incoming prompt fully contains the current resume frontier and has a following suffix, the suffix is prefilled from that frontier;
- when the incoming prompt ends exactly at a matching rewrite checkpoint, its complete continuation state is restored, and the same finalization unit produces the next anchor;
- when the incoming prompt matches a saved rewrite checkpoint and has a new suffix after it, after the atomic admission transaction commits, the growing allocations are truncated to the checkpoint frontier, the complete checkpoint is restored, and then the suffix is prefilled;
- when the incoming prompt matches a prefill context-checkpoint head at `F`, it takes `restore_context_checkpoint` or `restore_turn_rollback` (determined by the head kind): trim to `F`, load that head's GDN into current and its hidden into `tail_hidden`; for MTP set `mtp_kv_valid=F-1` and then take the existing bridge; for DFlash write that head's cyclic local back to that lane and set `dflash_context_frontier=F`. It is not `append_frontier`, and it does not read the rewrite hidden;
- a common prefix that ends at any other position without a checkpoint is treated as a cache miss.

Rewrite-checkpoint restore keeps the partial tail page that contains the checkpoint and releases the complete pages after it. A KV page or token prefix match is not by itself a checkpoint; the current architecture does not support arbitrary longest-common-prefix reuse.

Checkpoint kind is not a reuse compatibility bit. The Planner always first tries the current frontier by token, position, media identity, and complete continuation state, and then takes the longest complete checkpoint among the rewrite and that bundle's ladder heads; even if this request's `preserve_thinking` selects a different desired kind, a matching older snapshot can still be restored, and `prefix_reuse_path` reports the kind actually restored. The Planner takes the longest complete checkpoint among the rewrite and the ladder/turn-rollback heads; a current matching `E` still beats append; a same-`F` rewrite beats rollback/ladder. When `append_frontier` and `prompt_tokens > E`, turn-rollback is pinned at that `E`. An exact hit is not pinned by default; if the request asks for `capture_context_checkpoint` and that `E` has no head yet, it is written to the same slot. If the desired boundary lies in the suffix to be prefilled in this request, the physical slot is overwritten when it is crossed; if it already lies before the selected reuse frontier and there is no snapshot at that position, this request keeps the legal reuse and defers the new checkpoint, rather than proactively doing a full reset just to build an auxiliary snapshot. A full reset happens only when the incoming prefix matches no complete checkpoint.

A retained entry can be consumed by only one active request at a time. Multiple active requests do not share the same writable sequence state, and no copy-on-write branching is used. `cache_write: false` uses independent copied state and a protected cache source as described below. Request-local RNG, sampling, stop, generation limit, and output state are always created by the new request.

Retained state occupies actual state-pool memory but not an active control slot, and does not keep future growth reservation. Active admission has priority; before cache occupancy blocks an otherwise feasible request, the retained entries on free lanes are evicted first. VRAM retained physical state stays on its original lane, and the Planner selects the largest legal reuse among the free lanes. Ordinary consuming hits do not copy or migrate GPU page mappings; disposable forks copy payloads into independently owned mappings. Equal reuse prefers a free lane with no retained bundle; among equal-reuse dirty free lanes disposable state precedes ordinary state, then it covers the least-recently-admitted retained chat (monotonic occupy tick assigned after the first non-throwing `advance_prefill` in `start_prefill_lane`), then lowest lane index. An optional host-RAM second tier snapshots a completed bundle into pinned host memory at the admission site that is about to destroy it. A later hit restores the full SequenceState onto the chosen free lane and then uses the same prefix-reuse decision. Cache ownership is claimed only after the slot/lane and the full entitlement are already satisfied.

Prefix lookup changes only the uncached prompt work and the prospective reuse plan, and does not by itself grant queue priority. It may conservatively shorten the service projection of §5.5, but must still pass the same protected-head qualification; whether or not it hits, the final active request enters the same prefill/decode schedule and compact batch formation.

#### Disposable cache reads

`ExecutionOptions.cache_write` defaults to true. False selects a source-preserving fork, with source identity and destination lane planned independently. Longest preservation-feasible reuse wins, with direct VRAM, RAM, and disk tie preference; full-original RAM preservation is attempted when direct VRAM admission cannot coexist with its source. Reclaim unrelated idle retention before giving up. An infeasible longer hit does not exclude a shorter feasible hit or preserving cold path. Active lanes keep ordinary FIFO/deadline/head-protection semantics.

The family Program composes the selected current/checkpoint state into independent main/backend KV allocations, GDN state, hidden state, and DFlash cyclic state. Sampling and output control are request-local. Source mutation, page reclamation, and image/staging reuse wait through its readers; the single executor copy-hold owns initialization and cancellation. No page sharing or copy-on-write is used. The fork's durable disk association is cleared before execution.

An ordinary original may be captured to RAM when its lane or pages must be freed. This capture bypasses preserve-off rewrite cutting and preserves the complete original resume frontier plus existing checkpoints. Before eviction, failed/deferred admission can release the source claim and roll back that capture; after eviction, the saved original is committed and cancellation cannot remove its only image. A fork releases successful RAM restores instead of consuming them and keeps RAM-to-disk bookkeeping intact. Disk restore completion keeps the inclusive durable entry. A disposable original cannot be offloaded. Permanently impossible preservation on an idle Engine returns a request-local overload; temporary active/copy/reclaim pressure queues and replans.

False requests suppress future-reuse captures while retaining required execution transaction state. Their valid terminal state carries disposable retention. Empty eligible lanes are preferred; among retained victims disposable state precedes ordinary state, then admission recency and lane index decide. Apply the same order to destination replacement and shared-page reclamation. All capture entry points and both shutdown passes reject disposable persistence. A later true continuation sets ordinary retention for its resulting state.

Restore failure disposition distinguishes invalid source, stale plan, and destination resource failure. Only invalid source discards/quarantines the entry. Stale disk generation/claim races release pins and replan; destination resource failures keep a healthy source and can cold-prefill. Device-integrity failures remain Engine-wide. Request logs carry the write flag; saving an original is billed to its triggering request, without counting it as a disposable checkpoint capture.

### 6.5 Host RAM second tier

`--kv-ram-capacity` enables a startup-fixed pinned-host budget shared by optional completed-prefix caching and the secured live continuation images/credits of §5.8. The default is `off`. It changes active admission to bounded output windows but does not enlarge GPU pools or move Device Graph addresses. The remaining completed-prefix policy below concerns optional cache entries, not live images.

MTP or DFlash prefill may freeze current GDN into an Engine-wide staging slot during an in-flight prefill (after the Program prefill step compute-syncs that chunk). That freeze borrows slot `C` from the turn-rollback occupant: `occupied=false` before the clobber D2D, pack the ladder head, then H2D rollback GDN and hidden back. DFlash also D2Ds that lane's cyclic local into a 1-lane Engine-wide staging window before the same `d2d_done` fence; host-pack D2H reads that frozen window, not live local. The freeze is not a RAM capture; D2H of staging GDN and cyclic onto the live-lane host log may overlap the next chunk on the same lane. KV pages of the in-flight unit stay off `copy_stream`. `--no-prefix-reuse` disables ladder capture, turn-rollback pin, and restore.

Optional capture includes eviction bookkeeping: host allocation failure drops the capture and leaves inference running. Disk-assisted reclaim returns `Failed` and counts `optional_alloc` while preserving resident images. Live credit metadata allocation failure returns unfunded and preserves existing credits/images. Window grants reserve page-vector metadata before committing entitlements; failed allocation leaves the current window unchanged. Suspension metadata failure preserves the resident continuation. Resume metadata failure fences every reader before clearing only the partial destination and retaining the parked source. Failed synchronization remains fatal. Suspended-window planning reads stored page floors without allocating image descriptors.

Worker failure closes admission before cache-tier cleanup. `Engine::available()` reads admission state without waiting for GPU execution or tier cleanup. `/health` reports HTTP 503 while failed or stopping, and HTTP 200 for a healthy executor even when busy or full.

The completed-prefix cache is an exclusive FIFO of chats that are not on a VRAM lane. A new optional capture appends at the tail. Capacity pressure evicts the oldest unpinned optional entry until the image fits; if it still cannot, `drops` increments and execution proceeds without that optional image. Live-image backing is a precondition for RAM-on admission/window growth outside the completion cohort and is never silently dropped.

Capture and retained-lane eviction run only after `find_admission_lane` has already selected a free lane for that request. A queued prompt that still cannot fit — because in-flight lanes occupy enough shared pool pages that no free lane is feasible even after reclaiming every other free retained bundle — waits and does not dump any GPU lane. A later smaller backfill candidate that can fit may still admit and may capture other free retained lanes for itself.

The paragraph above governs optional completed-bundle capture only. A funded active suspension/replacement uses the separate §5.8 transaction and may release a settled blocked resident lane.

The executor captures at each admission site that is about to destroy a retained bundle, not inside `clear_lane` / `evict_retained_lane` / abort / destructor:

1. other free-lane retained entries the eviction loop is about to release;
2. a selected VRAM `FullReset` plan whose target lane is still retained (the usual `max_concurrency=1` load path);
3. a RAM restore that is about to cover a still-dirty target lane.

Capture queues D2H on `copy_stream` and **holds the source pages mapped** until `copies_ready`. Admit is two-phase: bind records the request in its lane and the retained victims awaiting release as copy-hold (including victims whose optional capture was dropped); other decode-ready lanes may run a DecodeRound while that D2H (and later restore H2D) is in flight. Admit-complete waits with `hipEventQuery` (and `hipEventSynchronize` on copy only when membership is empty), then `evict_retained_lane` / `kv.reset()`, optional restore H2D (a RAM restore starts only once its entry's own capture copies are ready, polled while others decode), `wait_kv_ram_copies_on_compute` immediately before this lane's `start_prefill_lane`, and harvest. Harvest of D2H/H2D elapsed happens after that wait, not on an overlapping DecodeRound launch.

A cancellation while other lanes decode does not drain at once: `begin_copy_hold_cancel` stops further disk restore work and records a fence on the copy stream behind the copies already queued, the hold stays parked while DecodeRounds continue, and the drain below runs once `copy_hold_cancel_settled` (or membership is empty), so it no longer waits on in-flight SSD reads. A shutdown with such a hold parked finishes it as cancelled. A `CacheRestoreFailure` before prefill consumes the prompt parks the hold the same way. Once it settles, cleanup releases the claims.

An invalid source is discarded and the request falls back to cold prefill; a stale plan preserves the source and replans admission; an unavailable destination preserves the source and falls back to cold prefill. Generation recovery keeps its lane and output while restoring or cold-prefilling the retry. It refreshes a stale host lookup once, then cold-prefills if that refreshed restore is also stale. Other request-local failures drain and fail only the request; other exceptions remain Engine-fatal.

If the held request is cancelled or fails before admit-complete, drain waits for those copies, harvests, releases unused RAM/disk claims, discards only sources classified invalid, and calls `evict_retained_lane` on every selected victim not yet evicted. A failed restore releases those victims the same way. If capture succeeded, the completed D2H image is the only remaining copy.

A later RAM hit claims the matching host entry (pinned entries are invisible to later `plan_match`). `capture` and `unpack` record a start HIP event before the copies and a done event after them so other-lane decode can overlap the DMA.

The pinned rewrite-checkpoint and ladder images (GDN conv/recurrent, about 150 MB per image, plus DFlash cyclic) copy host-to-host through `enqueue_host_copies`, a stream-ordered `hipLaunchHostFunc` callback in 4 MiB chunks, because ROCm runs `hipMemcpyAsync` between host buffers as a CPU copy inside the call. Each image's `copies_done` fence is stream-waited before and re-recorded after those copies; disk restore does the same on its state stream.

A RAM restore's installed ladder heads copy out of the entry on that host-copy stream behind only their own fences: no stream joins them and the entry's copy fence, which gates the lane's first prefill chunk, excludes them; a separate block fence keeps the entry's block allocated until they land.

Ordinary consume then erases that entry wherever it sits in the FIFO and retires the host block, including after an incomplete first chunk; a throw before consume releases the claim and leaves the host row in place. After consume the bundle lives only in VRAM until a later spill recaptures it.

Occupancy `used`/`entries` (human `kv-ram=` / `n=`) count those live host residents, including a claimed-but-not-consumed pin, and exclude retired copy blocks that still occupy the pin until reap. A later capture may still reap or evict while logged occupancy looks low.

The planner picks the larger `reusable_prompt_tokens` between VRAM and RAM; equal reuse keeps VRAM. VRAM `FullReset` and RAM restore both prefer a free lane with no retained bundle and cover a dirty lane only when no empty lane is feasible. Among those equal-reuse dirty lanes, the victim is disposable before ordinary retention, then the least-recently-admitted bundle and lowest lane index. Page-reclaim eviction of other free retained lanes uses the same order and never captures the selected lane twice. The admitted `RequestPlan` is the winner: RAM pass 1 keeps `evict_retained=false` even if the target lane is dirty, and restore captures that lane's old bundle. `GenerationResult` uses `prefix_reuse_source` for `none` / `vram_resident` / `host_ram` / `host_disk`; `prefix_reuse_path` still describes only frontier/checkpoint semantics.

An opt-in system-prefix LRU lives in the existing RAM arena (`--kv-ram-system-prompts N`, default 0). The family frontend emits a verified leading-system token frontier independent of the first user's content, including rendered tools and template instructions. A writable reusable request splits prefill at that frontier, captures all continuation state, and orders later compute after the entry's capture fence to protect active GDN/cyclic state and a partially filled last KV page. Restoring such an entry retains it and touches recency rather than consuming it. Its immutable prefix competes by the same longest-ready-prefix rules as ordinary RAM images. Count and arena-byte limits bound retention; normal FIFO/disk eviction cannot select system entries, and system snapshots are not durably spilled. Longer existing reuse is never sacrificed merely to capture an earlier missing system prefix. See `paged-kv-cache.md` for ownership and eviction details.

### 6.6 Persistent SSD tier

An explicitly enabled disk tier is inclusive beside the exclusive RAM FIFO and requires nonzero RAM capacity. Admission compares complete reusable frontiers, with equal reuse preferring VRAM, then RAM, then disk. For C=1, a disk restore does not enqueue its payload read until the retained VRAM victim has been evicted. Entries bind model/artifact identity and the exact AMD three-plane cache/checkpoint representation, including Vision-aware prefix identity; pool allocation capacity is not part of the durable fingerprint, allowing compatible restart resizing.

One restore coordinator and a startup-fixed reader pool own bounded pinned/device staging. A claim pins the immutable entry and pack generation. Validated page H2D/scatter can overlap later reads, but no partially restored lane is published. Admission waits for state restoration and copy-stream completion before prefill. Cancellation drains outstanding owners and releases the claim without consuming the durable entry. Restore reads yield to an emergency spill required for admission; idle spill uses separate staging and rechecks its epoch and RAM residency before publishing.

Durability publishes pack namespace, map, entry, then manifest with fsync ordering. Orderly shutdown captures retained VRAM, flushes nondurable RAM, waits for outstanding writes, and synchronizes the store before releasing lanes. Load-progress callbacks report the disk shutdown phases. Optional spill/allocation failure does not turn active requests into offloaded/preempted requests.

---

## 7. Scheduling model

These scheduling units and boundary rules apply in both RAM modes. Full-lifetime protected-head admission references select §5.2–§5.7 only with RAM off; RAM-on boundaries additionally resolve live transfers, retry window growth, and apply §5.8 before constructing the maximal funded membership. A growth-blocked or transfer-held request does not participate in a decode round.

### 7.1 GPU scheduling units

The Scheduler submits only three kinds of GPU compute work:

```text
PrefillChunk(request)
DecodeRound(all decode-ready requests)
MixedRound(all decode-ready requests, one prefill slice of the owner)   # DFlash only, §8.9
```

A complete request is not a scheduling unit. All **compute** GPU work executes serially on `device.stream`. Host-RAM D2H/H2D transfers run on `copy_stream`; they are not scheduling units, must not enter a Device Graph, and must not alias the pages of the current compute unit.

The Model Runtime owns one address-stable shared workspace, which is reused by the serial GPU units and is not duplicated per request. A prefill owner may hold one request-transient lease across the Vision/Text phases and multiple chunks; the lease is released after final prefill, cancellation, or failure. A copy-hold, like a prefill owner, holds the admission opportunity exclusively until that lane's copies permit `start_prefill_lane`.

### 7.2 Boundary processing

When the current GPU unit completes, or when a control event arrives while the GPU is idle, the GPU Executor performs one boundary:

```text
1. snapshot control events visible to this boundary
2. resolve the completed unit's per-request results
3. finish/cancel requests and release or retain their state
4. clean visible pending cancellation/timeout/permanent failures
5. if this boundary is an admission turn, apply one head-first
   protected admission/backfill opportunity (bind may return copy-hold
   without launching a GPU unit; complete starts this lane's prefill
   only after the relevant copies are ready)
6. choose, prepare and launch one next GPU unit
```

A parked copy-hold that was cancelled or whose restore failed is checked first: until its fenced copies settle, as long as the membership is non-empty, only that membership's DecodeRound runs, and generation recovery, admission, and prefill do not advance; only after they settle (or if the membership is empty) is the lane cancelled, or the failed entry dropped and the cold fallback taken. The remaining copy-holds are checked before the admission-turn gate: if the membership is non-empty and the copies are not ready, the DecodeRound of the other lanes runs first (no harvest, no `EventSynchronize` on the copy); if the membership is empty, the copy is `EventSynchronize`d and completed. Harvest happens only after that request's `start_prefill_lane` wait.

An event that arrives after the boundary begins is left for the next boundary. This guarantees that a membership update is finite and that continuous arrivals cannot delay the next launch indefinitely. The admission-turn gate, the frozen pending snapshot, and the protection state are defined in §5.6; a boundary that does not satisfy the gate only updates control/resource state. Cleaning up multiple invalid entries does not count as multiple admissions, but one boundary publishes at most one newly admitted request.

### 7.3 Decode/prefill policy

The scheduling policy is determined by two startup-fixed parameters: `mixed_forward=N` (the forward width of each prefill step while requests are decode-ready, a multiple of 256, where `0` means prefill-first; when unset, the planner selects it automatically: with DFlash and C>1, `N = min(1024, chunk)`, where chunk is `min(prefill_chunk, max_context)` rounded down to a multiple of 256; if that value is not greater than `(C-1)W`, or for any other backend, it is `0`) and `mixed_forward_rounds=D` (the decode rounds per slice, `D>=1`):

```text
if a parked copy-hold was cancelled or its restore failed:
    if its fenced copies have not settled and requests are DECODE_READY:
        run one DecodeRound containing them (nothing else advances)
    else:
        cancel it, or drop the failed entry and fall back to cold prefill
else if an admitted copy-hold exists:
    if its copies are not ready and requests are DECODE_READY:
        run one DecodeRound containing them
    else:
        complete it and start its prefill
else if a prefill owner exists:
    if N == 0 or no request is DECODE_READY:
        run the owner's next PrefillChunk (up to prefill_chunk tokens)
    else if the owner is mixable (§8.9):
        run D-1 DecodeRounds, then one MixedRound whose owner fills N columns
        less the round's live verify columns
    else:
        run D DecodeRounds, then one PrefillChunk of <= N tokens
else if a generation-recovery retry is queued:
    start it (a host restore parks it as a copy-hold)
else if a pending request is admissible and decode-admission budget permits:
    admit it and start its prefill
else if one or more requests are DECODE_READY:
    run one DecodeRound containing all of them
else:
    remain idle
```

With `N=0`, decode stalls for the prefill of the entire prompt, and prefill throughput is highest; `N>0` trades prefill throughput for decode progress. A mixed slice shares one weight stream with the decode rows, so at the same decode share the decode stall of a mixed round lasts only one round, while the stall of a separate `PrefillChunk` lasts the whole slice. The measured frontier is in [`performance.md`](../performance.md#mixed-prefilldecode-frontier-2026-09-29). When there is no decode-ready request, prefill chunks execute contiguously at the full `prefill_chunk`; when there is no prefill owner, decode rounds execute contiguously.

When there is no prefill owner and the ordered pending queue is non-empty, the head or backfill request selected by §5 takes the next prefill/finalization opportunity: when the GPU is idle it can be admitted immediately; when decode-ready rows already exist, one DecodeRound completes first, and then the selected request is admitted and its first prefill/finalization unit executes. If that unit does not complete, the request becomes the sole prefill owner and enters the policy above; if it completes, the request joins the decode batch at the next boundary. Sustained ingress therefore cannot admit multiple requests consecutively between two donor progress rounds, and cannot indefinitely delay the decode progress of the frozen frontier.

With `N>0`, the slice bounds the prompt work inserted between two decode rounds (or merged into one mixed round); with `N=0`, the decode stall covers the owner's entire prefill. When a decode-ready request exists, Vision encode forms a prefill step of its own (advancing no prompt tokens, `encoded_only`), and the subsequent decode rounds run before the text chunk that consumes it, so a decode stall never stacks encode and chunk. The specific token/media extent of a chunk/slice is a configuration qualified for the target and hardware, and is not part of the scheduler semantics. Vision and other prefill GPU phases must themselves constitute a bounded unit, or must already be counted in that chunk's latency bound; there is no unbounded prefill work outside the scheduler.

### 7.4 Joining and leaving decode

- A request joins the active decode set after final prefill/finalization completes; an exact retained hit skips only suffix work and does not bypass the existing prefill owner;
- A newly ready request first appears in the batch built at the next boundary;
- The protected-head policy affects only admission; once a backfill is decode-ready, it must not be excluded from the maximal batch in order to protect the waiting head;
- EOS, stop, or the generation limit removes the request after the completed round commits;
- The batch rows are re-compacted after every join/leave;
- An empty slot never produces an empty row.

### 7.5 Cancellation

Cancellation does not interrupt an in-flight GPU unit. The boundary observes cancellation once before launch, and a cancelled request does not enter the next membership. After the unit returns and before any per-row output preview/commit, one row-aligned cancellation snapshot is taken again against the frozen membership; this snapshot is unchanged for the whole resolve. A row cancelled at this point discards this unit's not-yet-committed result and closes the response output per the serving contract. The already committed continuation (KV, GDN current, rewrite checkpoint, ledger, and the context-checkpoint heads at or before that frontier) is retained for later prefix reuse, at the same boundary as `OutputLimit`. An unfinished suffix prefill rolls back to the occupy base: if a turn-rollback or ladder head exists there, that head is restored; otherwise, if a rewrite checkpoint is still held, it rolls back to that checkpoint. After the rollback, the staged heads beyond it are dropped and the lane's staging occupancy is released, and then the state is retained. When the occupy base is 0 and there is no rewrite, the whole `SequenceState` (including staged heads) is released. A speculative in-flight row folds with `commit_columns=0` and then retains the committed frontier from before the unit. Error/shutdown still releases the whole `SequenceState`. Provisional writes do not affect the other rows of the same round.

Therefore, cancellation waits for at most one GPU unit whose membership is already fixed, plus one boundary processing.

Ordinary decode is the exception to postlaunch cancel-retain: its in-flight step overwrites GDN/tail-hidden state without a snapshot of the preceding frontier. Such a cancelled lane is cleared rather than retaining a falsely rolled-back state. Cancellation observed before launch retains the committed lane and removes its request from the live batch. Speculative cancellation can restore its committed frontier, but invalidates the provisional tail-hidden value before reuse.

The cancellation of a waiting protected head or backfill candidate is removed in the §7.2 control cleanup; when the protected head leaves the queue, the epoch is atomically cleared, and the next-oldest entry does not inherit shadow state. The cancellation of a frozen donor shortens the frontier only after its active state is actually released; if that boundary is not an admission turn, the head retry is triggered first at the next legal turn.

---

## 8. Batched model execution

### 8.1 Runtime ownership

The resident Model Runtime is a model-instance object, not a request object. Within the Engine lifetime it uniquely owns:

- immutable model weights and the target schedule;
- the shared Sequence-State Store;
- one shared execution workspace;
- one `DecodeBatchFrame` of maximum capacity `C`;
- when a speculative backend is enabled, one all-layer ReplaySSM record arena of capacity `C`, whose fixed width is `draft_window+1` for MTP or the resolved `dflash_verify_width` for DFlash;
- startup-captured graph definitions and immutable profile executables.

Each request's persistent state exists only in the slot control and the `SequenceState` currently owned by that slot. The Model Runtime does not store a "current request", and does not duplicate the workspace, round buffers, or graph assets for each slot.

A decode round additionally creates two short-lived objects:

| Object | Location and lifetime | Content and ownership |
|---|---|---|
| `RoundMembership` | host; from batch build until that round's commit completes | immutable `B` and the `row -> slot` mapping; keeps the slots/state alive |
| `DecodeBatchFrame` | host/device stable storage; reused serially by all rounds | typed input controls, state selectors, round activations, and result staging; owns no sequence state |

`RoundMembership` does not upload request identity. The device schedule sees only the typed selectors required by the target/Op; completed results are mapped back to the original slot by the host mapping. While GPU work is in flight, the membership, the related slot bindings, the state allocation, and the frame ingress must not be modified.

### 8.2 DecodeBatchFrame

`DecodeBatchFrame` is planned once at Engine startup for `C`. Each exact-`B` schedule obtains only prefix views of the same backing, and no separate set of round memory is allocated for each `B=1..C`. Stale rows beyond `B` are not read by the graph, so the boundary does not need to zero the inactive tail.

The typed fields required by ordinary decode include at least:

```text
current_tokens[B]
cache_positions[1,B]
RoPE positions/deltas[per target contract, B]
Main KV table rows[B] + optional MTP KV table rows[B]
lanes[B]
SamplingConfig[B] + logical sampling positions[B]
sampled_tokens[B]
```

`lanes[b]` is the stable execution lane of row `b`. Qwen3 uses it to select both the Linear Attention current state and the continuation-hidden destination; speculative Fold also obtains the same lane from the frozen membership. The KV table row remains separate, because it describes a paged allocation binding, not fixed-state ownership.

Hidden activations, mixer intermediates, and logits use exact-`B` views in the same shared frame/workspace. `DecodeBatchFrame` is the runtime's logical collection of these typed regions, not a generic descriptor ABI passed to all Ops; each Op still receives only its own semantic inputs. Host/device control staging may transfer the entire small SoA frame of fixed capacity `C` in one transfer; the model schedule reads only exact-`B` prefixes and submits no model work for the inactive tail.

Batch assembly handles different data differently:

| Data | Batching method |
|---|---|
| Small controls such as token, position, and sampling config | written per compact row into contiguous batch ingress |
| KV payload and block tables | kept in the shared paged pool; only the per-row table-row selector is written |
| Linear Attention / backend fixed state | kept in the shared state pool, located by stable lane or backend-defined row |
| Request stop, output, and external identity | kept only in the host slot; does not enter the model graph |
| Activations, hidden, logits | produced in shared execution memory by one whole-batch schedule |

KV or recurrent state must not be gathered/copied to build a contiguous batch, and a device-pointer array must not be built for each row. The final hidden or other continuation image that the target needs to retain across rounds must be published to the respective `SequenceState` through typed destination selectors within the same batched schedule, and not through a separate D2D copy per request at the boundary.

### 8.3 Round preparation

After the boundary selects a `DecodeRound`, the Target Batch Assembler prepares in the following order:

```text
all DECODE_READY slots
        │ compact once
        ▼
immutable RoundMembership[B]
        │
        ├─ materialize each row's already-reserved one-round state growth
        ├─ fill current token, exact positions and typed state selectors
        ├─ fill per-row sampling inputs and continuation destinations
        └─ select one whole-batch execution profile
        ▼
publish exact-B ingress -> launch
```

Every member of an ordinary round has one valid column, so there are no inactive rows or valid-column masks. Materialization may update block tables for different allocations, but the pool/table-matrix base addresses remain unchanged. All rows must be fully prepared before launch; it is not allowed to launch part of a batch first and then supplement state for another row.

Profile selection uses a bound over the whole batch, for example the largest visible-context frontier among all rows. Exact per-row positions and context lengths remain device data. If a legal active set cannot be represented by a single profile, then that route does not satisfy the concurrency contract, and the rows must not be split into cohorts.

The host may run a lightweight loop over the metadata for `B<=C`, but GPU ingress must be a batch-level publication; an independent scalar copy, transition kernel, or model call must not be issued for each row.

### 8.4 Ordinary round state transaction

A `DECODE_READY` sequence always stores a target-defined committed cursor and a single current decode anchor. Taking the Qwen3 ordinary transition as an example:

```text
before:
  target state valid for positions [0,p)
  current anchor token is at position p

execute:
  process the B anchors at their respective positions
  write each row's provisional KV/recurrent/continuation state
  sample one next anchor per row

commit for surviving row b:
  state frontier  := p[b] + 1
  current anchor  := sampled_tokens[b]
  anchor position := p[b] + 1
```

An Op writes only the physical state permitted for this round and does not own the committed frontier. The graph also does not advance a compact-row position or selector in place into "next-round state"; the authoritative cursor is updated by `SequenceState` at the boundary commit, and the next round generates controls from the new row mapping. This lets rows move freely, without vectorized scalar helpers or persistent row-local state.

An ordinary round licenses exactly one token for each uncancelled row. EOS, stop, or the generation limit can make that token a terminal token, but the model state and the output of the same row are not cut at different frontiers. Cancellation is the only ordinary boundary outcome that can discard a whole row's provisional result; that `SequenceState` is then released.

Qwen3 ordinary GDN calls the existing width-1 Snapshot leaf with `initial_state_slots=lanes` and `snapshot_base_slots=lanes`. After fully reading the row's initial checkpoint, the leaf overwrites the same current slot in place, so it produces no speculative trajectory and needs no additional state slot.

### 8.5 Whole-model execution

When the logical batch size is `B`, the model schedule of one replay is:

```text
current_tokens[B]
  -> one batched embedding/input stage
  -> one traversal of all model layers
  -> publish B continuation images
  -> one batched lm_head
  -> one batched sampler
  -> sampled_tokens[B]
```

The ordinary column layout is equivalent to `[D,1,B]`, and column-independent projection/MoE/FFN routes consume the aggregate `T=B` at once. Control code may iterate over lightweight row metadata, but must not call a layer, model schedule, lm_head, or sampler separately for each request.

Per-request KV/context, recurrent state, sampling state, and output ownership remain independent; batched execution merges the current activation columns and the weight-consuming work, and does not concatenate multiple sequences into one sequence.

### 8.6 Operator contract

| Operator family | Batched execution requirement |
|---|---|
| Linear, projection, dense FFN | one operator schedule consumes the aggregate row/token extent |
| MoE | route over the aggregate token set, then execute grouped experts over the whole set |
| Full attention | consume per-row context length and the substrate-provided KV view as a ragged batch |
| GDN / linear attention | execute one batched transition over independent per-row recurrent-state handles |
| `lm_head` | produce, at once, the logits of all active rows or verification positions |
| Sampler | consume per-row sampling configuration and RNG state, without splitting the batch |

Different context lengths, stop conditions, sampling values, or generation limits do not form different model batches. A request option that requires a different model topology must be fixed at Engine startup, or else is rejected as unsupported.

The specific compact layout, typed extents, and selector ABI are defined by the corresponding `include/ninfer/ops/` semantic contracts; this architecture does not duplicate the per-Op interface list.

### 8.7 Result resolution and commit

Token logprobs follow the same rule. A request that set `OutputOptions::top_logprobs` raises its row's `logprob_rows` flag in the round's control frame. Every decode schedule (ordinary, MTP, and DFlash, captured or eager) ends its token decision with `ops::token_logprobs`, which scores each produced token against the logits column that decided it (column `i` for the supported chain verification) and writes a fixed-size record per token slot into the result frame: the token's log-probability and the 20 highest-ranked ids with theirs. The launch is part of every definition, so no graph variant or post-replay launch exists; a row whose flag is clear returns from the kernel before reading logits and its record slots are not written. The prefill-sampled first token uses the same Op on the scalar step logits, before the MTP bridge reuses them. The Engine turns the records of the committed token prefix into `TokenLogprob` values per output channel; tokens that publish to no channel are dropped there.

After graph replay, only the compact per-row result is returned, for example the ordinary sampled token; logits or per-layer state must not be returned. The GPU Executor waits for one whole-round completion, then resolves each row independently through the frozen `RoundMembership`:

```text
row result
  -> locate its still-owned slot through RoundMembership
  -> apply cancellation / stop / output-limit policy
  -> commit the exact licensed prefix to SequenceState
  -> update request sampling, usage and owning output record
  -> mark DECODE_READY again or MODEL_FINISHED
```

For each row, the model-state frontier, target cursor, sampler/penalty progress, usage, and owning output record form one logical transaction. An output event may become visible to the response path only after all of this state is committed. A row finishing, being cancelled, or accepting a shorter prefix in speculative mode does not change the commit result of the other rows.

A round sampled with model stop tokens excluded may commit only through the first completed `</tool_call>`. A plain chat round also commits through its reasoning close before sampling the continuation with the dedicated closing token excluded. Tool grammars enforce the close boundary in their per-node matcher masks and reserve closing-think markup only in free prose; constrained argument values retain literal markup. `resolve_pending_batch` folds the licensed prefix.

When the speculative backend's target GDN uses ReplaySSM, the GPU graph only reads the lane's current state and writes the Program-owned raw records; it does not advance the committed GDN state. After the CPU output preview obtains the final commit length of each row, `resolve_pending_batch` first executes one all-layer Fold with the original `B` rows (except continuing DFlash chain rows; see deferred Fold below), then completes the necessary hidden/backend correction, and advances the host frontiers only after synchronization succeeds. A cancelled row participates in the original row mapping with `commit_columns=0`, the Fold is strictly a no-op for that row, and then the row's already committed continuation is retained, rather than releasing the bundle. The Executor may commit the output preview and publish the output event only after this commit tail succeeds.

After all rows have resolved, the `RoundMembership` is destroyed and the frame may be overwritten by the next unit. Slots that continue running are re-compacted at the next boundary; no row identity is inherited from the current frame into the next frame.

### 8.8 B=1 and prefill

`B=1` uses exactly the same membership, frame, whole-model schedule, and commit transaction, and no separate request-local ordinary decode path is kept.

Prefill remains a single-sequence unit. It exclusively owns its `SequenceState` but reuses the Model Runtime and the shared workspace; it does not hold a persistent `DecodeBatchFrame` row. After final prefill establishes the complete decode cursor, the request joins ordinary decode only at the next boundary, through normal batch assembly.

### 8.9 Mixed prefill/decode round

With DFlash, a mixable owner's slice shares the target forward of a decode round. The owner is mixable while it is Prefilling a text-only prompt: no Vision/media phase and no MTP bridge preparation. Other owners use the separate-step path of §7.3. A tool-grammar owner's final step mixes too: its first-token root mask occupies the exchange's last row, which a verify batch (at most C-1 rows while a request prefills) never reaches, and computing it leaves the batch's bindings intact (Speculative grammar exchange).

```text
columns = [owner slice (S_eff tokens) | verify rows (B x W_live)]
S_eff   = min(remaining prompt, N - B x W_live)      (at least N - (C-1)W before the last slice)
```

The forward width is fixed at the configured `N`, a multiple of 256 (prefill holds its rate at 256-multiple widths and loses 3-5% between them). The owner takes the verify columns a round leaves unused, when fewer than `C-1` requests decode or adaptive draft length gives `W_live < W`, so the aligned forward width holds in every round.

- Linears, norms and MLP run once over all columns, so the weights stream once for both the owner's prompt tokens and the verify rows.
- Attention runs per segment: the owner's KV transaction and causal prefill attention first, then each verify row against its own cache. Verify KV status is indexed by lane.
- GDN: the owner uses the prefill convolution and chunked recurrence on its own current slot; verify rows use the record path over their column range.
- DFlash feature capture is split between the owner's prefill sink and the verify taps. The owner's DFlash context append runs after the round's egress. The owner stages its ingress in frame slot `C-1`, because the owner is never a member and `B <= C-1`.
- Final owner steps finalize and sample as in ordinary prefill.
- Admission with decode-ready requests and a configured mixed forward does not run a mixable owner's first step: `start_prefill_lane` returns without progress, and the first slice joins the next round, so decode never stalls for a separate first step. That empty step spends no service quantum and does not restart the `D` cadence.
- The owner's KV transaction is committed as the round's last host action, after the owner and verify tails, the egress copy and the owner context append are enqueued. Its status read therefore waits on the finished round instead of stalling submission between the last layer and the tails.
- Capacity: the plan rejects an `N` that is not a multiple of 256, exceeds the prefill chunk, or cannot hold `(C-1)W` verify columns plus one prompt token. The workspace plan reserves a separate `dflash_mixed` layout of `N` columns with an owner of up to `N - 2B` columns at each batch `B` (verify panels are at least two columns wide).
- Mixed rounds are eager. Pure decode rounds keep their Device Graph definitions, and adaptive-draft round-time observation skips mixed rounds.

The owner's prefill Ops and state transitions are those of prefill-first; greedy owner tokens matched prefill-first in every pair check, and 8K PPL is identical at chunk 2048 and 256. Verify rows take the wide-T Linear/norm routes of the mixed width. They are checked against the same Op oracles, but they are not bitwise equal to a pure decode round, so greedy streams may diverge at near-ties. The owner's KV commit synchronizes at the end of the round, so a mixed step's prefill timing and the decode round time both span the whole mixed round.

---

## 9. Device Graph model

### 9.1 Exact-B graph definitions

Because `C` is small and fixed, the Engine captures an exact definition for every logical batch size at startup:

```text
Definition[family, B=1, profile]
Definition[family, B=2, profile]
...
Definition[family, B=C, profile]
```

`family` is fixed by the Engine startup mode as ordinary, MTP, or DFlash; all three modes use exact `B=1..C`, and DFlash keeps no separate `B=1` concurrency route.

`B=1` is a first-class exact path. A single request never runs a graph permanently padded to `C` rows. Each definition represents the complete decode round for its semantic family, exact `B`, and whole-batch profile, and binds the same shared frame, workspace, and state-pool bases. It reads typed controls and selectors, not request/slot identity.

Each reachable exact `B`, captured K and context profile owns one immutable executable:

```text
Definition[family,B,K,profile] -> Executable[family,B,K,profile]
```

Definitions keep their rewritten device storage alive for their executable's lifetime. The finite profile inventory is clipped to the configured context ceiling before capture. There is no executable update or shared installed-profile state: repeated HIP executable updates retain kernel-argument allocations and cannot be assigned a finite runtime budget. Different request identities still use the same immutable executable for the same B/K/profile.

The startup preparation order for a graph family is fixed:

Loaded FP8 Linear descriptors must already cover the full prefill chunk, ordinary widths `B=1..C`, and every flattened verification width `B*W(K)` of the frozen captured-K set. Load planning and Program binding consume the same family width authority; preparing only the maximum storage width is insufficient for adaptive K3/K4/K5. Descriptor preparation does not occur inside stream capture. The maximum workspace extent remains separate from this finite descriptor inventory.

1. For each enabled semantic family, run one eager round with the first reachable profile at `B=1`, so that HIP code and the library runtime finish lazy materialization before stream capture;
2. Capture all exact-`B`/profile definitions. Capture only records HIP work; it does not execute a model round and does not carry production numerical qualification;
3. Instantiate and upload every profile's executable. For each family, exact batch, captured K and target structural class, smoke-launch the profile with the largest frontier. Prepare its frontier and synchronize before each launch. Structural class is startup warmup metadata; executable ownership remains one per profile;
4. Reset all mutated state and controls before exposing the Program. Capture's temporary repeated private-page mappings remain valid through every smoke launch.

Startup materializes the full immutable executable inventory. Smoke is a resource/lifetime check; independent operator tests and real-artifact integration remain the numerical authorities.

### 9.1a `B=1` persistent lowering

After capture and before instantiation, `ops::persistent_decode_lower` rewrites the `B=1` definition in place: every run of at least two consecutive hosted kernel nodes in the graph (small-T LQ4 / row-scaled FP8 projections, FP8 activation producers, GDN front / pair-conv / record, Q/K norm-RoPE) is replaced by one child graph containing a cooperative persistent kernel. That kernel stays resident with two 384-thread blocks per WGP, treats each captured launch as one phase, and executes it with its original kernel body (for GDN record, a per-(sequence, head) re-tiled body of the same Op; at widths >8 it remains a kernel node), its original arguments, and virtual block/grid indices, with a grid barrier between phases; the output is bitwise identical to that of the original kernel nodes. The remaining nodes (verify attention, drafter, sampling, copies) are unchanged, and the graph remains a single chain.

Each persistent run is captured with `hipLaunchCooperativeKernel` inside an owned child graph. Dispatch is cooperative; occupancy alone is insufficient for a kernel with whole-grid barriers. The installed ROCm runtime stores the graph-node cooperative attribute separately from the dispatch mode, so setting that attribute alone is insufficient. The child graph preserves the actual cooperative command and its dependency ordering when cloned into the definition. Capture failure affects only the temporary graph, never the owning definition. The capture stream and temporary graph are released before replay. Startup adds a 20-MiB allowance, based on observed additional cooperative graph runtime residency, including calibrated bulk configurations; this is charged once per model instance rather than once per executable.

Before an ordinary, MTP, or DFlash `B=1` executable launches, Program joins already-submitted disk page-scatter events onto the copy stream, then enqueues compute dependencies on the fixed prior load and copy frontiers. This prevents transfer kernels from competing with the resident grid: cooperative dispatch alone reproduced an unentered final block during cache-pressure inference on gfx1201. Disk state H2D already joins the copy stream. These edges wait only for submitted GPU work, never for future SSD reads, and introduce no host synchronization. The Engine worker owns GPU cache submission, so it cannot enqueue later transfer work during that round. Existing compute-to-copy and prefill load-release events precede these new edges; no edge refers to a future decode round. `B>1`, mixed rounds, and uncaptured execution retain transfer overlap.

Lowering interns byte-identical run images (phase records, captured arguments, and a grid-barrier word) in startup-owned, fixed-address 2-MiB arenas. Each `DecodeGraphDefinition` retains the arenas its nodes reference, keeping storage alive through its immutable executable's lifetime. The startup index and host copies disappear after preparation; replay performs no lookup or allocation. Shared images require the Engine's one ordered compute stream: kernels sharing a barrier word must never overlap. Uploads finish on the null stream before a nonblocking compute stream can consume them. `B>1` definitions retain individual kernel nodes (lowered C4/C8 was slower); see `docs/performance.md`.

### 9.2 Dynamic active set

When the active set changes, the runtime publishes a new frame ingress and selects the matching pre-captured definition:

```text
[A]       -> Definition[B=1]
[A,B]     -> Definition[B=2]
[B]       -> Definition[B=1]
[B,C,D]   -> Definition[B=3]
```

Runtime selects the already instantiated executable for the exact B/K/context profile and replays it once. Active identity changes only typed ingress data; no request, slot subset or bitmask enters the graph key. Only the GPU Executor launches executables. Serving never calls HIP graph instantiate, update or upload.

Paged-KV allocation, Linear Attention state ownership, and retained-state reuse change only block-table content or selector values. A graph always binds the pool/table-matrix base, so these ownership transitions do not form a graph key either.

### 9.3 Context and prefill profiles

The exact context length is a typed runtime control value. If the exact target needs a small number of different kernel topologies for different context ranges, then besides `B`, a finite set of whole-batch profiles fixed and captured at startup may be used. A profile is selected once by a batch-level bound, for example the maximum row context-length bucket; the graph still executes ragged work with the exact per-row lengths.

A context profile must not split the active set into cohorts, and must not be keyed by request identity or active-slot combination. A profile that cannot simultaneously represent arbitrary legal per-row lengths cannot serve as a concurrent decode route; a route that preserves whole-batch execution must be used.

Prefill is single-request work and may use a target-specific fixed execution profile. It does not change the decode-batch graph model, and it must not trigger, during serving, a graph capture that would block active decode.

### 9.4 Stable execution memory

All exact definitions reuse:

- one `DecodeBatchFrame` planned for `C`;
- one workspace sized to the maximum over all reachable phase/`B`/profile peaks;
- the same set of model/state pool bases;
- one set of address-stable pinned host ingress/result staging planned for `C`.

A graph-enabled exact-`B` definition contains one batch-level control upload at the start and one batch-level result download at the end. Staging objects may be transferred at the fixed size `C`; only the first `B` rows have semantics for the current round:

```text
fill pinned ingress rows [0,B)
        -> optional already-reserved KV page/table materialization
        -> one graph replay:
             one H2D typed-control frame
             whole-model batch schedule
             one D2H result frame
        -> one round completion wait
```

Cross-page materialization happens on the same execution lane before replay, does not change the captured pool/table bases, and does not form a graph key. Graph-off mode submits these actions eagerly in the same order and, like graph mode, executes the transaction at the maximum execution frontier of the selected planned profile (that bound determines the attention grid and workspace extents and the DFlash envelopes), so eager execution and captured replay are bitwise identical. Packed decode attention selects each sequence's context partition on the device from the batch size and its live row positions, so both modes read the same partition from the same device data; a host-side live frontier would instead form a different launch and break graph/eager exactness.

Logits, hidden, workspace, or per-sequence state must not be duplicated for each `B`, profile, or captured definition. There is no per-row device submission in model/control ingress, forward, and result egress; table publication when crossing pages belongs to state substrate materialization. No graph family is captured, instantiated, or extended during serving. The startup reservation includes all reachable definitions, their owned lowered storage, and one immutable executable per definition. Ordinary and uncalibrated DFlash selectors use a conservative family allowance plus the measured per-executable allocation bound. MTP reserves its measured inventory on the device-argument-pool path: 60 MiB of family residency (capture-time storage, lowered images and the warm launches) plus one 4-MiB executable per definition, or 6 MiB once the exact-B executable's argument pool crosses the next 2-MiB step, from B=12 at K3, B=9 at K4 and B=8 at K5; K1 and K2 stay at 4 MiB through C16. Executable size depends only on K and B, not on the context profile, KV codec, proposal head, sampler or Vision, and adaptive MTP sums its captured Ks (gfx1201, HIP 7.16.26385, 2026-11-13). Production C1..8 uses a driver-policy-specific measured full-inventory reservation documented in `docs/performance.md`. The low-residency calibration requires all four settings: `DEBUG_HIP_GRAPH_CLASSIC_PATH=0`, `ROC_ENABLE_LARGE_BAR=0`, `ROC_SKIP_KERNEL_ARG_COPY=0`, and `HSA_DISABLE_FRAGMENT_ALLOCATOR=1`. Immutable arguments reside in pinned host memory; cached dispatch packets replay in bulk with `DEBUG_HIP_GRAPH_BATCH_SIZE=16384`. The same preset without those settings retains the conservative device-argument-pool bound; other presets use the full inventory bound. These HIP startup settings do not change the graph inventory or the mathematical execution schedule. The measured C6/full-context/adaptive-K7/chunk1024 profile also uses the 62 MiB bulk-argument allowance; other chunk1024 selectors retain the conservative inventory bound. The observed-allocation startup guard remains mandatory; eager execution reserves no graphs. NInfer-induced lazy driver resources belong in NInfer's reservation. The configurable KV headroom is reserved exclusively for external GPU users and display activity.

The temporary Paged-KV allocation used by capture/smoke materializes only one private physical page per valid row, and publishes that page repeatedly into the temporary block-table row. Preparing one eager/smoke run zeroes only the temporary pages, fixed-state lanes, and typed controls corresponding to `[0,B)`; definition capture itself only resets the host allocator of the shared workspace. The whole physical pool must not be zeroed according to the configured KV capacity, and the state of all `C` rows must not be zeroed for each definition.

---

## 10. Speculative decoding

Speculative decoding is an Engine-level decode mode:

```text
speculative_backend = off | MTP | DFlash
```

The R9700 product takes DFlash/DFlash2 as the preferred speculative path, whose support and optimization must be completed. MTP remains a supported backend; migrations and later changes must continue to protect its existing graph/eager semantics, cache, row view, accept/commit state, and tests, but no new MTP feature, head-precision, or performance-optimization work is assigned, and such work is not a release gate.

All requests of the same Engine use the same backend. The Scheduler still submits only one `DecodeRound(all decode-ready requests)`.

Without adaptive drafting, the proposal window is startup-fixed and only that K is planned and captured. Startup-enabled adaptive drafting preplans the bounded supported K set (DFlash {3..N} for `--draft-tokens N>=5`, MTP {3,4,5}), with separate K-specific graph definitions and accounted memory. At each round boundary one K is selected for the whole compact batch.

Each row's remaining output/context budget constrains its logical proposals and publication, not necessarily DFlash's physical captured K. DFlash can choose a larger padded K only when its cost has been measured at the current concurrency and its full physical width fits every row's remaining context. K3 remains eligible; no new graph or storage is created. Existing masked fallback remains at the context boundary.

The policy uses observed conditional acceptance and measured round time at concurrency/context length to estimate tokens per second; it does not split acceptance cohorts or capture graphs during serving. A smaller live K does not shrink the backing storage or discard previous full-width pending DFlash features.

DFlash selects at the actual round boundary after publication establishes the compact batch. Expected yield is clipped to each row's logical extent (including one guaranteed target output for a zero-draft active row), while timing and K histograms describe the physical route. Hop acceptance is observed only for actual proposals. MTP retains its existing budget-clamped policy. These are functional scheduling semantics, not a claim that adaptive K beats a physically qualified fixed policy on a particular artifact.

R9700 DFlash is chain-only, K<=7, W=K+1<=8 including the target anchor; no packed-tree or two-block product schedule is exposed. Production runs adaptive K{3..7}: 7-8-row / 32-token drafter and GDN routes plus two-tile small-T FP8 target Linears keep C1..C4 rounds off the prefill GEMM (2026-09-28, `docs/performance.md`); fixed-budget W8 root-sibling/best-first trees remain rejected. The DFlash companion also does not automatically inherit the fixed Q4 recipe: it independently compares canonical Q4G64, source-MSE Q4G64, and source-MSE W8G32 from the real BF16 DFlash2 checkpoint, and adds row-scaled E4M3 only when physical small-width speed and quality both support it. Every recipe keeps the BF16 selector codebooks and private BF16 state.

Internally this round executes:

```text
all pending request anchors
  -> one batched proposal phase
  -> one batched target verification
  -> per-request accepted-prefix result
  -> per-request state commit
```

MTP advances proposal positions autoregressively, and each position executes as a batch over all active requests. DFlash produces the proposal block through one batched block forward. The difference between the two exists only inside the selected decode graph and does not change slots, admission, scheduling, or active-batch formation.

MTP's Text verify and backend alignment/AR cache writes both use round-scoped segmented typed transactions. Each compact row has its own fixed-address device status/cursor; the ingress base frontier initializes the cursor, and the pool-wide block-table base together with the device row selector selects the actual sequence mapping. Alignment and each AR segment advance the cursor only after their all-layer append completes; there is no D2H or host publication mutation inside the graph. After graph/eager execution completes, the Program resolves each row's status/cursor exactly once: Text keeps the complete verify suffix, and MTP publishes only the licensed alignment frontier; any invalid count/position/table mapping poisons that row's publication, and it must not be partially published.

MTP and DFlash keep their own proposal frame, proposal state, and graph topology; they share only the non-owning target verify/accept view, the per-row accepted-prefix result, and the target state transaction. The concurrency layer neither builds a virtual interface that hides backend differences nor maintains two sets of membership or commit loops for the two backends.

Engine capability is fixed at startup. An MTP Engine can enable Vision at the same time; when MTP's shifted input lands on a vision column, it uses the same Vision-composed embedding as the target, and the prefix-reuse bridge follows the same input semantics as normal prefill. DFlash also accepts Vision-conditioned target features. Its proposal cache positions remain logical text positions; target verification applies each lane's MRoPE position delta independently. This does not add a Vision encoder to the draft checkpoint or change the private BF16 draft cache.

Different requests can accept different numbers of proposals. Acceptance length is result metadata; it must not be used to split verification, replay the model, or form an acceptance cohort. The target model is always the output authority, and only accepted target/backend state can commit.

Qwen3's MTP and DFlash share the same target ReplaySSM transaction: target verify writes each layer's convolution/key/value/gate records into a fixed arena per compact row, and the physical record row always equals the batch row of this round; after the CPU obtains the final output prefix, one Fold uses the frozen `lanes[b]` to commit row `b` to the current state of that lane. Rows must not be compacted or reordered because of cancellation or different acceptance lengths. Records lie inside the Device Graph, and Fold lies in the eager commit tail after the CPU decision; the next GPU unit must wait until the current records have been consumed by Fold before it can overwrite the arena.

Deferred Fold: when a DFlash chain row commits `committed > 0` and is not terminal, `resolve_pending_batch` does not fold it in the commit tail, but records `(committed, record row)` in `SequenceState`; that lane's GDN slot stays at its round-base state (none of the round's columns applied), while ledger/KV/hidden advance as usual. The next DFlash round writes these rows into `DFlashDecodeIngress::gdn_fold` (per row of that round, including the old record row and slot), and target verify first applies each GDN layer's Fold for that layer. When aggregate tokens >32, the batched chain uses `gdn_replay_prepare_layer` to first publish the convolution history and save the old committed K/V/gate prefix into the caller workspace; the front may then overwrite the old records, and `gated_delta_net_replay_record` completes the Fold when it first reads the FP32 state, publishes the committed state, and then executes this round's not-yet-accepted recurrence, saving one state read. Smaller forwards and diagnostic taps that read recurrent state keep the Fold fused inside the FP8 front; other paths use `gdn_replay_fold_layer`. Both approaches consume before overwriting, and the result is bitwise identical to the all-layer Fold. Lanes not in the next round, tree rounds, and Program entry points that read or capture lane state (prefill, retain, capture/restore, score, shutdown) first flush eagerly with the all-layer Fold; `clear_lane` discards an unapplied Fold.

From the completion of target execution to the end of Fold, the request is Pending: the authoritative execution/ledger frontiers, the ledger contents, and the prefix identity still stay at the round base; licensed tokens and backend staging exist only as unpublished candidates. Fold, Text and optional MTP KV visibility, continuation hidden, proposal-backend continuation, and the host frontier must commit the same final prefix. A continuing row commits all licensed outputs; a terminal row may commit a strict prefix because of stop/EOS/output limit; a cancelled row commits zero columns and releases its sequence.

Each row's valid proposal extent is bounded by the remaining output/context capacity. Unused positions in the fixed proposal window are masked and must not update request state or output.

If a row is not yet terminal but its valid proposal extent is zero, the same DecodeRound must perform ordinary target progress for that row; an active row that does not advance must not be kept. EOS, stop, output limit, or context limit truncate the effective committed extent before the per-row commit, and model state and output commit only the same effective prefix.

---

## 11. Completion and response handling

Ingress establishes an owning record for the request; the effective output bound is determined at admission. Each boundary appends the committed tokens to that record, so neither an ordinary nor a multi-token speculative round depends on network progress to commit.

When the model completes, the GPU Executor finalizes the owning result record, releases the request's slot and unused reservation, and either releases the sequence state or, per §6.4, transfers state that has target-declared reusable checkpoints to the retained-prefix cache. After the above GPU resources are reused, the response path can still continue to read the owning record.

Committed output events are written to request-owned result/event storage and are taken out by the caller thread waiting on that request. Network writes, protocol serialization, and `OutputSink` callbacks do not run on the GPU Executor. A slow caller does not block the GPU boundary loop; the backlog is still limited by that request's finite output bound. Request lifetime capacity continues to be charged until the response consumer releases the owning record, so the reuse of the model slot does not create an unbounded completed-result backlog.

Model completion and response completion are therefore two independent lifetimes:

```text
model lifetime:     admission ───────────── model finished
response lifetime:  request received ───────────────── response finished
active slot/state:   admission ──────────── model finished
retained state:                                 optional ───── eviction/reuse
```

A GPU execution or state-integrity failure is an Engine-wide failure: admission stops, active and queued requests end with unavailable/failure, and serving can resume only after the Engine is recreated. The Scheduler does not guess that some row of a failed shared round can safely continue.

Target selection reports nonfinite eligible raw logits in lane-owned, per-verification-column status words. The Program clears them before each round and validates every live target column before returning generated tokens. Suppressed tokens do not set status, and status from inactive columns does not affect the publication decision. A reported nonfinite target logit is a state-integrity failure under the Engine-wide policy above. MTP proposal argmax returns an in-domain placeholder when no ordered maximum exists, keeping device gathers safe while the host awaits target validation. Immutable Device Graph replay uses the same status storage and publication checks as eager execution.

---

## 12. End-to-end examples

### 12.1 New request joins an existing decode

```text
time ───────────────────────────────────────────────────────────►

GPU: Decode[A]
       │ boundary: admit B as the prefill owner
       ▼
     Prefill[B, chunk 0]
       ▼
     Decode[A]
       ▼
     Prefill[B, final chunk]
       │ B becomes decode-ready
       ▼
     Decode[A,B] -> Decode[A,B] -> ...
```

B does not run a separate decode forward; instead, after its final prefill it joins A's next logical batch.

### 12.2 Two rows assembled and committed

Suppose A and B each occupy one of two slots, and hold completely different context/state allocations:

```text
                         row 0 (A)        row 1 (B)
current token               101              202
cache position               41              900
Text KV table row             5                1
stable lane                   0                2
sampling state lane           0                2
```

The Batch Assembler writes only the above controls/selectors into the frame. KV pages and Linear Attention state stay in place in the shared pools. `Definition[B=2]` performs one complete model traversal over the two columns and returns:

```text
sampled_tokens = [303, 404]
```

The boundary uses the frozen mapping to hand 303 to A and 404 to B. If A continues and B's 404 is a terminal token, both rows first commit their own correct model/output frontier, and then B's slot/state is released or retained. The next round re-forms the batch as A's `B=1` frame; A's KV row and stable lane do not change because the compact batch row changed.

### 12.3 Active batch grows and shrinks

```text
ready requests          selected definition

[A]                     Definition[B=1]
[A,B]                   Definition[B=2]
[A,B,C]                 Definition[B=3]
[B,C]       A finished  Definition[B=2]
[B,C,D]     D joined    Definition[B=3]
```

Neither request identity nor the occupied slot affects graph identity.

### 12.4 Shared 128K context with two slots (RAM off)

Suppose the relevant shared context capacity is 128K units:

```text
request A: used prompt 50K + reserved output 30K = 80K commitment
request B: prompt       32K + reserved output 16K = 48K commitment
                                                       -----
                                                       128K
```

A and B can be admitted at the same time. As context grows, reserved capacity converts to used capacity, so the `used + remaining reservation` of the two never exceeds 128K; it can reach the boundary but cannot cross it.

If B needs `32K prompt + 32K output = 64K`, only 48K remains after A's commitment. B stays in the pending queue and becomes the protected head, until A releases capacity, B's queue timeout expires, or another terminal event occurs. NInfer does not admit B first and then decide which active request to truncate when approaching 128K.

If there is also a later request C, it can backfill only after passing the full vector accounting of §5. Explained on the Main capacity dimension alone, A is B's frozen donor; beyond the capacity that A's release leaves for B, there is still 64K of shadow surplus. As long as the other dimensions such as lane/optional-MTP also hold, this gives C a frontier-safe upper bound; but C must also satisfy the current `fits-now`, and the real free space at this moment is only 48K, so C can currently obtain at most 48K. Even if C is still active when A completes, the release boundary still admits B first. If C needs to borrow B's critical capacity, it must also satisfy the temporal work credit, and a frontier miss triggers drain.

### 12.5 Persistent-safe and temporal backfill (RAM off)

The following example writes only the Main entitlement; the actual decision checks the lane and optional MTP pages dimension by dimension at the same time. Suppose the capacity is 174K and `C>=4`:

```text
active A: 64K, projected to finish first
active B: 64K
free:     46K
head H:  100K, exclusively feasible but blocked now
```

After selecting `D={A}`, the frontier accounting is:

```text
B 64K + H 100K = 164K
safe surplus = 10K
```

An 8K candidate P can become a persistent-safe backfill:

```text
B 64K + P 8K + H 100K = 172K <= 174K
```

Even if P is still running when A releases, H retains its full entitlement. Another 20K candidate T fits into the remaining capacity at present, but cannot join the persistent-safe ledger; it can serve as a temporal backfill only if its complete prefill/decode service work is projected to finish before A's release frontier and the epoch temporal credit is sufficient.

```text
Decode[A,B]
  -> admit/fill P
  -> Decode[A,B,P]
  -> admit/fill T if temporally qualified
  -> Decode[A,B,P,T] ...
```

If A reaches EOS early while T has not yet finished, the Scheduler does not interpret this estimation error as a new backfill opportunity:

```text
A releases
  -> retry H first
  -> H is still blocked by T
  -> PROTECTED_DRAIN
  -> no new admission; existing P/T and B continue
  -> T releases
  -> admit H
```

If T has already finished in time, or the early completion of another active request makes the exact accounting sufficient, H is admitted directly at the corresponding boundary. Later arrivals never became donors of H, and cannot restore temporal credit that has already been consumed.

---

## Related documents

- [Paged KV context storage](paged-kv-cache.md)
- [ReplaySSM GDN technical reference](replayssm-gdn.md)
- [Serving behavior](../serving.md)
- [Qwen3.8-27B model semantics](qwen3.8-27b-model.md)

### Idle maintenance and scoring ownership

Idle KV copy readiness and spill submission run under execution ownership after rechecking the queue and active slots. The queue mutex is released before any Program or HIP call. If scoring holds execution ownership while slots appear empty, the worker skips Program access and retries after a bounded wait. A short idle-maintenance mutex serializes this host submission section with `score_many` admission, so maintenance alone cannot make an otherwise idle Engine report overloaded. Scoring releases that mutex after reserving execution ownership; normal decode never takes it.

### Constrained output and grammar exchange

The Qwen frontend owns immutable compiled tool/response schemas, EBNF, full-match regex and literal choices, and transactional matcher state. `PromptOptions::output_constraint` constrains answer content without injecting schema text into the prompt. Response schemas enforce assertions with JSON Schema defaults and fixed property order; unsupported assertions fail preparation. Reasoning remains unconstrained until `</think>`. Constrained content preserves literal special-token bytes and EBNF whitespace, while model stops keep ordinary publication policy. Tool declarations and response constraints cannot be combined; tool-call parsing is enabled only for tool grammars. The existing HIP `TokenMaskExchange` supplies ordinary root and speculative chain masks, and only committed publication advances matcher state. Caller stops and limits can truncate a valid prefix.

`OutputSession` previews tool lifecycle events with content and reasoning, publishing only the accepted target prefix. The executor assigns stable request/call identities. Safe first-string argument prefixes stream; ambiguous or constrained values wait for schema-guided decoding. Syntactic envelope completion remains provisional until the terminal generation result accepts the call. Cancellation and incomplete envelopes create no executable history. Requests eligible for duplicate-tool recovery hold tool events until an attempt is accepted, discarding them on retry. Transport adapters own protocol framing, defer accepted completion events as needed, and preserve budget/cancellation finish reasons.

Constraint observations read committed complete/terminated/branch state. Compilation and compiler-lock wait are nested host elapsed times. Matcher counters count each computed mask once, including speculative positions, but exclude unreachable dummy rows; upload counters measure the restricted payload after shared-row compression. Round synchronization and the exchange mutex provide the existing ownership handoff, with no added device synchronization. Recovery adds prior-attempt work once; terminal state and cache identity describe the accepted/current attempt.

`TokenMaskExchange` owns a fine-grained coherent host mailbox and a matcher thread; the round's Device Graph contains no host node. Once the verification IDs exist, a publish kernel on the compute stream copies IDs, parents and valid-column counts into the mailbox and raises its request number with a system-scope release. The matcher, armed from before the round's launch until `finish_round()` after its synchronization, polls the request (20 us), runs the grammar and posts the sampling configurations, the restricted mask rows and a release of the reply number. Rows whose masks permit the whole token domain carry a null mask; a row whose nodes share one mask publishes it once with column stride zero.

Target verification then runs embedding through LM head, and an acquire kernel before the masked argmax waits for the reply and copies the configurations and restricted rows into the planned device buffers, so the host match overlaps the target forward.

Acquisition waits for the owned matcher without an elapsed-time deadline: valid grammar work and host scheduling delays must not disable the Engine. The matcher posts a reply even when matching throws, using nonempty fallback domains so the round can synchronize; `finish_round()` then reports the error before publication. Representative graph warm replays use the same arm/synchronize/finish lifecycle as live speculative rounds.

Ordinary root masking uses its synchronized CPU boundary. A prefilling request's root mask is written into the last row without rebinding, at admission and again before its final step (another prefilling request may have written the shared row since); inside a mixed round that host fill precedes the runner's enqueue, so the armed matcher is still idle.

Bindings change at synchronized round boundaries, which the matcher cannot overlap: it holds the exchange mutex while answering. Lane release and Program teardown retain the owning execution boundary before request/OutputSession storage is released. Graph-branch alternatives are not available on this HIP: a side-stream branch executes serially, segment scheduling makes rounds 2.3x slower, and a captured external event wait crashes (`docs/performance.md`, Decode step attribution).
