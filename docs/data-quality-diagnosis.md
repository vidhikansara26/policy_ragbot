# Incident: stale Privilege Leave (PTO) entitlement

Two-Question Debugging Framework applied to the planted data-quality fixture.
Evidence below was captured with `python -m rag_lab query` and `answer`.
`tests/test_data_quality_diagnosis.py` pins every claim on this page.

## Symptom

```
$ python -m rag_lab query --k 1 "How many Privilege Leave / PTO days do I get?"
Query: How many Privilege Leave / PTO days do I get?
Indexed 116 chunks from 10 documents.
Returned 1 chunks.

Chunks
1. score=6.406  Human Rights Policy  v1.0  legacy  section=3. Fair Wages and Remuneration

Answer
Human Rights Policy, section '3. Fair Wages and Remuneration', version 1.0 (status=legacy)
  Full-time employees are entitled to **15 days of Privilege Leave (PTO)** per calendar
  year.
```

The assistant published **15 days** as the answer. Coforge's current Human Rights
Policy does not publish a Privilege Leave day count at all. The number is real,
but it is retired.

## Question 1 — did retrieval find the right documents? **Yes.**

```
$ python -m rag_lab query --k 5 "How many Privilege Leave / PTO days do I get?"
1. score=6.406   Human Rights Policy  v1.0  legacy   section=3. Fair Wages and Remuneration
2. score=0.106   Supplier Code of Conduct  v2025  current  section=Labor Management and Human Rights
3. score=-1.023  Human Rights Policy  v2.0  current  section=5. Fair Wages and Remuneration
4. score=-3.999  Human Rights Policy  v1.0  legacy   section=Preamble
5. score=-8.315  Policy Against Sexual Harassment at Workplace  v2024  current  section=11.0 Action During Pendency of Inquiry
```

Both versions of the Human Rights Policy are inside the top five: the legacy
v1 clause at rank 1 and its current v2 counterpart at rank 3. The retriever
did not lose the current policy, mis-embed the query, or drop a chunk. Recall@5
for this question is 1.0, and the eval harness labels the row
`data_quality_fixture` rather than `miss`, because retrieving the planted
clause is the incident the index exists to surface.

Ruling out the pipeline, stage by stage:

| Stage | Could it explain the wrong answer? | Evidence |
|-------|-----------------------------------|----------|
| Chunking | No | The v1 clause and the v2 counterpart are each intact in one chunk with correct `section` metadata |
| Dense (MiniLM) | No | Both versions are retrieved; the query embeds against both |
| BM25 / RRF | No | Fusion returns both versions inside the candidate pool |
| Cross-encoder | No | It ranks v1 first *because v1 is the only passage that actually answers "how many days"* — that is correct relevance behaviour |
| Source data | **Yes** | v1 states a number; v2 states that no number exists |

The cross-encoder score gap is the tell. v1 scores `+6.406` and v2 scores
`-1.023` on the same question. That is not a ranking defect: the question asks
for a day count, and only the retired document contains one. A relevance model
cannot infer that a factually responsive passage is administratively void.
Recency and lifecycle are properties of the corpus, not of relevance.

## Question 2 — did the system use the right one? **No.**

The extractive `query` command cites rank 1 verbatim, so it published the v1
clause. The current v2 passage was retrieved and then ignored, because nothing
in the extractive path compares `status` across versions of the same document.

## Root cause: source data

A retired document was indexed with no supersession relationship to the
document that replaced it. `data/raw/human_rights_policy_v1.md` carries
`version: "1.0"`, `status: "legacy"`, and the 15-day clause;
`data/raw/human_rights_policy_v2.md` carries `version: "2.0"`,
`status: "current"`, and the wording "does not set a numeric Privilege Leave
(PTO) entitlement". Both are indexed, which is deliberate — see
`config.FIXTURE_SOURCE_FILE` and `docs/corpus.md`.

This is the realistic enterprise failure: nobody deletes the old policy. It
stays on the share drive, gets crawled, and outranks the current version on
the exact question it was written to answer. No amount of retrieval tuning
fixes it, because retrieval is behaving correctly.

## Why the stale document stays indexed

Deleting v1 would hide the incident rather than resolve it, and would destroy
lineage. An enterprise assistant must be able to answer "what did the policy
used to say, and when did it change?" for audit and grievance review. The
platform therefore keeps v1 and resolves the conflict at answer time, where
the resolution is visible and citable.

## Remediation

1. **Retain lineage at ingest.** `status=legacy` is never filtered in
   `corpus.load_corpus` or `index.PolicyIndex.upsert`. Version and status are
   required chunk metadata (`config.REQUIRED_METADATA_FIELDS`).
2. **Prefer current sources at generation.** `safety.screen_hits` orders
   `status=current` ahead of `status=legacy` before the prompt is built, keeps
   the current counterpart of a surviving legacy hit even below the relevance
   floor, and `generate.generate_answer` abstains outright on a legacy-only
   window rather than publishing a retired clause.
3. **Block the retired fact from prose.** The leak guard in
   `generate._publishable_prose` drops a sentence that is supported only by a
   legacy passage, so `15 days` cannot reach a published answer.
4. **Mark the conflict in citations.** When the same `doc_name` appears as both
   current and legacy in one window, the legacy citation is rendered with a
   `(legacy conflict)` suffix instead of being hidden.
5. **Keep the incident measurable.** The eval harness scores the row as
   `data_quality_fixture`, and `conflict_handling` in `evaluate_answers` fails
   if a generated answer publishes the legacy day count.

## The score floor, and why it needed an exception

Remediation steps 2 and 4 did not originally reach the CLI path for this
question. `config.MIN_RERANK_SCORE` is `0.0` and the current v2 counterpart
scores `-1.023`, so `screen_hits` removed the current policy before the prompt
was built and left a window of the retired clause plus an unrelated supplier
passage:

```
screened abstain: False | legacy_conflict: False
  kept +0.106  Supplier Code of Conduct  v2025  current
  kept +6.406  Human Rights Policy       v1.0   legacy
```

The leak guard still blocked `15 days`, so the retired number was never
published. But with v2 screened out the window no longer held both statuses of
one document, `has_legacy_conflict` was `False`, and the answer could only
abstain while citing v1 without the `(legacy conflict)` mark.

This is the same root cause seen from the other end. The floor assumes score
tracks usefulness, and on a version conflict it does not: the retired clause
scores high *because* it answers the question literally, and the current clause
scores low *because* it says the figure no longer exists.

`safety.screen_hits` now applies a conflict-companion rule. When a legacy hit
survives the floor, the highest-scoring `status=current` hit sharing its
`doc_name` is re-admitted even from below the floor. The companion still has to
pass the lexical gate, so the window widens by version lineage only, never by
relevance, and a window where nothing clears the floor still abstains outright.
`tests/test_safety.py` pins all four behaviours.

The CLI now answers from the current policy and marks the conflict:

```
$ python -m rag_lab answer "How many Privilege Leave / PTO days do I get?"
The current policy does not set a numeric Privilege Leave (PTO) entitlement.
Leave entitlements for Coforge employees follow local law and service
conditions, as stated in the current Human Rights Policy and Supplier Code of
Conduct.
Sources:
- Supplier Code of Conduct, Labor Management and Human Rights, v2025
- Human Rights Policy, 5. Fair Wages and Remuneration, v2.0
- Human Rights Policy, 3. Fair Wages and Remuneration, v1.0 (legacy conflict)
```

`15 days` is still absent, v1 is still indexed, and the retired version is now
visible beside the policy that replaced it rather than silently dropped.

## A second prompt-level finding

Closing the floor gap exposed one more defect, in the prompt rather than the
corpus. The instruction read "Set grounded to false when no current passage
supports the question." Asked how many PTO days the policy grants, the model
correctly observed that no current passage states a day count and therefore set
`grounded: false`, discarding its own accurate answer:

```
{"grounded": false, "answer": "The current policy does not specify a numeric
Privilege Leave (PTO) entitlement. Your entitlement follows local employment
laws and service conditions."}
```

The instruction made "current policy sets no such figure" unanswerable by
construction: the model had to report an absence while being told an absence
means ungrounded. `grounded` now describes the answer rather than the question,
and reporting that current policy sets no figure counts as grounded when a
current passage says so. This is worth recording next to the data-quality
incident because it is the mirror image of it — here the retrieval and the
corpus were both right, and the instructions were wrong.
