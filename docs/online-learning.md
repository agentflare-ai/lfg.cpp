## Agentic Architecture

lfg.cpp is not a general-purpose inference wrapper. It is a production agentic inference engine
built around a core insight: the model's own internal signals — entropy, confidence, surprise —
are first-class routing primitives, not debugging artifacts.

### Monitoring as Agentic Routing

The entropy, confidence, and surprise monitors drive real-time decisions during generation:

| Signal | Condition | Action |
|---|---|---|
| Surprise monitor | High surprise on input | Store input in vector DB — this is new knowledge outside model priors |
| Entropy monitor | High entropy during generation | Retrieve from vector DB — ground the uncertain span before generation continues |
| Confidence monitor | Sustained confident span | Skip retrieval — model already knows this, avoid polluting context |
| Structured decode failure | Grammar constraint violated | Heal from last valid checkpoint — do not restart, preserve KV cache and reasoning context |

This means the vector DB is not a static knowledge base consulted on every call. It grows
exactly where the model's uncertainty signals say it should, and is consulted exactly when the
model signals it is in uncertain territory. The model routes its own memory operations.

### Structured Output Healing and Checkpointing

Standard structured output implementations restart generation from scratch on constraint
violations. This burns tokens and destroys the KV cache, losing accumulated reasoning context.

lfg.cpp checkpoints the last valid grammar state during constrained generation. On failure,
generation resumes from the checkpoint rather than from zero. The model retains its context;
only the constraint violation is unwound.

### Continuous LoRA Adaptation

The same signals that trigger vector DB operations are the training data signal for continuous
LoRA adaptation.

**Training data collection:**

A high-surprise input that led to a successful task outcome is a gold-labeled training example.
The model encountered something outside its distribution, the vector DB grounded it, and the
agent completed the task correctly. That full sequence — input, retrieval context, output — is
a demonstration worth learning from.

**Adaptation loop:**

```
surprise on input → store in vector DB
high entropy during generation → retrieve from vector DB
task completes successfully → (input, retrieval, output) triple is a training example
LoRA trains on accumulated examples → model priors shift toward the domain
same inputs now score lower surprise → retrieval fires less often → inference is cheaper
repeat
```

**Self-validating metrics:**

The same monitoring signals that generate training data measure whether the LoRA worked:

- **Surprise rate decreasing** — model has internalized the domain, priors are better calibrated
- **Retrieval rate decreasing** — model generates confidently from weights, not from external memory
- **Decode failure rate decreasing** — model has learned the structured output patterns for this task class

No held-out eval set or human raters are required. If the LoRA is good, the production signals
get quieter. If it makes things worse, they get noisier and the adaptation is rolled back.

**Guard against confident wrongness:**

Inference metrics improving is a necessary but not sufficient condition for promoting a LoRA.
A model can become confidently wrong — low surprise, low retrieval, clean structured output,
but incorrect results. This failure mode does not appear in inference metrics; it appears in
downstream outcome quality and rework rates.

LoRA promotion requires inference metrics improved **and** outcome quality held or improved.
Training on inference signal alone risks training into a confidently wrong model.

### Cost as a Unified Signal

Fewer retrievals and fewer decode failures means fewer tokens per successful outcome. LoRA
improvement is directly visible as lower token cost per task class. Inference quality and
operational cost point at the same number.