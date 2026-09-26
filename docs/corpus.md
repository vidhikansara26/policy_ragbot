# Corpus: source-document compliance

Primary sources are public Coforge investor policies from
[investors.coforge.com/policies](https://investors.coforge.com/policies).
Markdown under `data/raw` is the indexed form. PDFs are not stored in git;
each file keeps a `source_url` pointing at the investor PDF.

## Roles

Every document has a `role` in YAML frontmatter. Loader rejects any other value.

| Role | Files | Why |
|------|-------|-----|
| `primary` | POSH, Whistleblower, EHS, Human Rights v2 | Assignment set: four current policies, 500–800 words |
| `fixture` | Human Rights v1 | Stale data-quality incident (15-day Privilege Leave). Length is not gated. |
| `supplemental` | Supplier CoC, Nomination & Remuneration, Modern Slavery, Board Diversity, CSR & ESG | Extra investor PDFs that still hold eval gold (80% training pass rate, Independent Director terms) |

Do not filter `status=legacy` at ingest. Do not merge v1 into v2.

## Word counts (body only, whitespace tokens)

Frontmatter is excluded. `body_word_count()` is the same function the tests use.

| File | Words | Chars | `##` sections | Role |
|------|------:|------:|-------------:|------|
| `posh_policy.md` | 657 | 4,288 | 9 | primary |
| `whistleblower_policy.md` | 737 | 5,186 | 7 | primary |
| `human_rights_policy_v2.md` | 688 | 4,734 | 13 | primary |
| `ehs_policy.md` | 596 | 4,636 | 7 | primary |
| `nomination_remuneration_policy.md` | 468 | 3,194 | 8 | supplemental |
| `supplier_code_of_conduct.md` | 411 | 3,168 | 5 | supplemental |
| `csr_esg_policy.md` | 388 | 2,777 | 5 | supplemental |
| `modern_slavery_statement.md` | 344 | 2,430 | 6 | supplemental |
| `board_diversity_policy.md` | 337 | 2,368 | 5 | supplemental |
| `human_rights_policy_v1.md` | 246 | 1,677 | 6 | fixture |
| **Total** | | **34,458** | | 10 documents |

Primary band: **500–800 words**. POSH was already in range and was not rewritten.

## Preprocessing

1. Download the investor PDF named in `source_url`.
2. Extract text (PyMuPDF). Drop cover pages, tables of contents, version-history grids, and the closing “About Coforge” marketing page.
3. Rebuild markdown headings from the PDF section numbers so citations can name a real section.
4. Keep role-based mailboxes (`shrc@coforge.com`, `whistleblower@coforge.com`, `All.HR@coforge.com`). Personal named inboxes from the PDF are not copied.
5. Preserve eval gold spans character-for-character. Do not paraphrase them.
6. Length-align only the primary set: expand Whistleblower and EHS with extra **procedure** from the same PDF; trim Human Rights v2 **non-eval** sections. Do not invent a current numeric Privilege Leave / PTO entitlement. v2 states that the policy does not set one.

Human Rights v1 is a reconstructed stale copy (NIIT Technologies branding, `hr.helpdesk@niit-tech.com`, 15 days of Privilege Leave). The public FY 2025 PDF does not publish a day count. v1 stays indexed so retrieval can surface the data-quality incident.

## Gold spans that must stay unique

These strings are labels in `src/rag_lab/eval.py`. Each must remain an intact substring of exactly one file.

| Span | File |
|------|------|
| `The Sexual Harassment Redressal Committee email id is **shrc@coforge.com**.` | `posh_policy.md` |
| `within three months` | `posh_policy.md` |
| `whistleblower@coforge.com` | `whistleblower_policy.md` |
| `acknowledge receipt within **5 working days**` | `whistleblower_policy.md` |
| `Net zero by 2040` | `ehs_policy.md` |
| `The central EHS Committee reviews the policy annually.` | `ehs_policy.md` |
| `does not set a numeric Privilege Leave (PTO) entitlement` | `human_rights_policy_v2.md` |
| `15 days of Privilege Leave (PTO)` | `human_rights_policy_v1.md` |
| `pass rate of 80% or over` | `modern_slavery_statement.md` |
| `two consecutive terms of up to a maximum of 5 years each` | `nomination_remuneration_policy.md` |
