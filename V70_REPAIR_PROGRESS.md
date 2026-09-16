# V70 repair - final local handoff, not deployed

## Final state (2026-09-14 14:54 KST)

This section supersedes all partial checkpoints retained below. No benchmark or API verification process remains running. Source/data connection and local verification are complete for this repair scope; the all-requests-95-point objective is NOT achieved.

- Whole paired run completed all 1,600 scheduled calculations, 400 requests per product/version, zero execution errors. Baseline perfume9/400 → candidate18/400; baseline lotion132/400 → candidate164/400. No formerly passing request lost its pass. Score regressions and no-score cases remain explicitly reported, not removed.
- Final artifacts: `.benchmarks/v70_repair/full400-paired/final-comparison.json`, `.benchmarks/v70_repair/output-invariants-final.json`, `V70_FULL_SCOPE_RESULTS.md`. Final output checker inspected every raw response; candidate invariant violations0, baseline diagnostic zero-percent rows2 retained.
- Candidate09: all43 API routes covered by117 calls/21 checks passed; public integration21 calls/all3830 unique coverage rows passed; regulatory57,450 checks plus158 full-pool sums passed; installed wheel and relocated private bundle validated.
- Duplicate-free regression union:1,772 passed, zero unresolved failures/errors/skips. This combines the full suite, affected retests and optional dependency supplements; it is not one single command.
- Public source consent and acquisition completed. Final352 source documents, source bundle SHA `495b8d1276a5bd9b77f36bf6ad3ab4f575712dd6bcccbad563bb3aa6b82a534d`. Public prices243/3830; missing stock/lead-time/operator evidence remains missing, never fabricated.
- Private release prepared: `tmp/modal-runtime-v70/release-01`, bundle SHA `f94ea18d1035fb8112f1cb21265448311eec909466aa7d4d94994325792cfb23`, wheel SHA `22767b37f437e92306423782e63cee88834b8ac91b130f504a36cb38941375bd`. See `MODAL_DEPLOYMENT_V70_PREPARED.md`.
- Final Git hygiene: all88 workflow-referenced source/test paths exist and are not ignored; newV70 regression tests are explicitly included. LocalV70 bindings/raw source data/private deployment bundle remain ignored. Scoped credential-pattern scan of30 new source/handoff files found no token/private-key matches. No Git staging, commit, push, PR mutation, Modal deployment, Linux image build or key change occurred.
- Final reader-facing handoffs: `V70_FULL_SCOPE_RESULTS.md`, `AI_REGULATORY_DATA_V70.md`, `MODAL_DEPLOYMENT_V70_PREPARED.md`; README updated with full denominators and local-only status.

## Historical execution notes (partial results below are superseded)

Latest user scope: fix all discovered failures, source actual regulatory/supplier data, verify IFRA/EU REACH/K-REACH/FDA integration, and assess the entire active catalog and fixed 400-case corpus rather than examples.

No production deployment, GitHub push, or existing-key change has been performed in this repair turn. Deployment requires user approval. The actual live service remains V69 Modal v17. V70 is a software repair label; the shared trained V69 checkpoint is unchanged. The user's later explicit permission to agree to terms was applied to ECHA CHEM's public terms, not interpreted as deployment or permission to evade restrictions.

## Current evidence

- Original full audit: `.benchmarks/modal_v69_full_audit/summary.json`.
- Original live flaws: perfume examples 61-85 points; woody lotion 94.8400 twice on cache misses; English timber intent lost; cold language worker about 78 s; 30 local test failures.
- Fixed English wood vocabulary and protected negation/word boundaries.
- Added full-vector cutting planes with full-pool dual column pricing to lotion optimization. Final score/physical checks unchanged.
- Added analytic neural-score gradients to perfume dose optimization, preserving the prior neural floor.
- Added conservative cap/cost fractional-LP dual upper bounds. Clean/citrus/woody has a rendered optimistic upper bound about 85.96 under current profiles and constraints; this is a model bound, not human perception.
- Restored 19 original historical artifacts verbatim from the prior workspace, preserving hashes. One source mismatch was exactly a missing final blank line, now restored to its recorded hash.
- Fixed incomplete mocks and historical/current-release test contract mismatches without changing old numerical outcomes.
- Full local suite: `full-tests-01.xml`: 1,708 tests, 1,707 passed, 1 PostgreSQL skip, no failure/error. The skipped test was subsequently run on an isolated PostgreSQL 16.15 localhost fixture and passed; cluster stopped. `postgres-test.xml`, `postgres-runtime.json`.
- Later framework tests: 73 passed (`framework-tests.xml`). Later cache/framework/model tests: 62 passed (`cache-framework-tests.xml`); overlapping suites, do not add counts blindly.
- All 3,830 active materials x 450 fine-head outputs compared exactly equal between old and cached identity path. Warm fine-view stage 0.843 s -> 0.024 s; not total API latency (`all-material-cache-equivalence.json`).

## Global data acquisition

`scripts/expand_all_public_materials_v70.py` inspected the complete PerfumersWorld public catalog (1,212 products), joined all 3,830 active materials by exact CAS, excluded explicit diluted stocks from neat-material matching, and fetched all 257 matching SKU document pages.

Result: `.benchmarks/v70_repair/public-evidence-all-01/coverage_summary.json`:

- 243 active materials with public prices/source documents; 3,587 unmatched, explicitly retained in `all_active_coverage.json`.
- 261 archived source documents; 3 failed direct official fetches (IFRA index/amendment and ECHA HTTP 403). European Commission REACH, Korean official law and FDA official guidance were archived successfully. Supplier IFRA limits are specimen/source observations, not full legal review.
- Public bundle SHA256: `0fee4e893513a2e5e8c087c606ea65cecabbc4b04482cdd9c656f3c592941f40`.
- No quantitative stock, binding quote expiry, confirmed lead time, operator registration or review is invented.
- Public data store is separate from trusted operating evidence. `/v2` operational recommendation remains blocked without operating evidence. Explicit `diagnostic_only=true` permits calculation with public evidence but never recommendation/approval. True previous public snapshots support observation change comparison.
- Regulatory tabs consume public source findings and separate actual registered framework checks. New `/v2/evidence/status` and paginated `/v2/evidence/coverage` expose whole-population coverage and missingness.

## Global comparison currently running

`scripts/full_scope_benchmark_v70.py` runs all 400 unchanged bilingual cases for perfume and lotion in both original deployed package and candidate-04: 1,600 actual local API calculations total. Four CPU processes, fixed wheels extracted independently. Every response is stored compressed; failures and unsupported requests remain in the denominator.

Output: `.benchmarks/v70_repair/full400-paired/`.

Poll `summary.json` for progress and `results.jsonl` for raw rows. Current tool session at time of writing: 28522. Do not mutate this runner while it is active: its resume protocol pins the runner hash. Do not replace or overwrite old results. Last observed 437/1,600; incomplete, so no overall improvement claim yet.

Baseline wheel: `039bb684df56908263473e85f7b7e69e26f72e2717c3d4812bc151b6c0aedc07`.
Numerical candidate-04 wheel: `9c91552e1a5421f571a477831a2d6163240b934279414d8b874cde97ca46a127`.
Shared model: `3186e467446abc71f249c052c8248c17c97d0e337976abf3666628af47dd997f`.
Catalog material digest unchanged: `8c962601b6567d56bd1043733b2b2eaccf9deb5d35e4aba70101271fe65bf139`.

Candidate-04 numerical probe (`probe-05/report.json`): peony/apple 97.6685; woody lotion 95.1082 (5.73 s); cedar/sandalwood 89.2276; clean/citrus/woody 70.4395; rose/apple 87.5790; vanilla/amber 82.2430; bodywash 87.1866. These are examples, not the final global comparison. Never claim all requests reached 95.

Candidate-05 added whole public data/framework coverage. Candidate-06 added exact-output identity caching and IFRA blocked-status precedence. Candidate-08 is the current final source-bound local build; its scientific numerical code equals candidate-04 except the previously verified exact-output identity cache. Public regulatory source indexes and serialization changed. Do not turn off source guards.

## Latest whole-scope data and verification (2026-09-14)

- Candidate-06 full API audit completed: 117 calls, 21 checks, all passed; all 43 routes covered. `all-api-06/report.json`.
- Candidate-06 actual public data integration: 21 calls passed, all 3,830 coverage rows retrieved. `public-api-06/report.json`.
- User explicitly authorized ECHA terms; accepted through the official browser UI. Official offered downloads succeeded (no WAF bypass): restriction list 79 entries / 238 history rows, candidate list 253 entries / 264 history rows, authorisation list 59 entries, REACH registrations 25,370 rows, harmonised CLP 4,822 rows.
- Korean official export: 47,520 records, all streamed despite the XLSX incorrectly declaring A1:A1 dimensions. Exact CAS 943 active materials; an additional 230 matched through PubChem structure-corroborated CAS aliases. These are not operator registration evidence.
- PubChem: queried all 2,937 candidate source CIDs in 74 bounded public batch requests. Exact stereochemical structure equality confirmed for 2,937 materials, no canonical CAS overwritten. All 3,830 identities remain in the denominator. Among 1,370 missing-CAS materials, 994 have corroborated CAS aliases; 804 have a unique alias. 34 materials have no source structure, 859 have no exact external corroboration.
- IFRA official 51st overview: all 263 standards. Complete 726-page standards PDF and 73-page guidance extracted. All 1,887 numeric product limits for all 12 supported product categories verified against individual PDF sections, zero discrepancy. Do not claim final 52nd rules (consultation update only).
- IFRA other-source contributions: 1,046 usable natural/Schiff-base rows. One source typo `0..3` preserved as unresolved; its parent CAS has no active material match. No guessed correction.
- Four runtime indexes: IFRA, EU_REACH, K_REACH, FDA. FDA uses all 11 summary headings, but named/class screening is not exhaustive CFR/MoCRA, safety or labelling review. Explicit unresolved scopes are returned.
- Global runtime verification: 57,450 material/framework/product checks plus 158 full-pool group sums, passed. `all-regulatory-08.json`. This is software verification, not 57,450 real-world safety approvals.
- Final public bundle: `public-evidence-registry-03/public_evidence.json`, SHA256 `ea3a29aaee3355448643017c336c70dff3111c67e125973d4140460f722b1519`, 347 source documents. All 3,830 examined, scoped source matches IFRA 226 / EU 858 / KR 1,173. Public supplier price still 243; no fake stock, quotation or business evidence.
- Final candidate-08 wheel SHA256 `f944ddcd26c42a232ec917ac02c2f67596d24147201f7f6f0c9f47f20cd3299c`; profile `perfumery.v70-candidate-08.local.json`; preparation at `.benchmarks/v70_repair/candidate-08/preparation.json`.
- Ongoing final API audit session35384 (`all-api-08`), actual source connection session24366 (`public-api-08`), full final regression suite started. Do not change fragrance_ai source while these profile-bound audits are running.
- Whole paired benchmark session28522 still running; last observed 1,049/1,600. Source packages are immutable and unaffected by current regulatory additions. Never call partial results final.
- `summarize_full_scope_v70.py` produces paired full-denominator scores, gains/losses and failure reasons. `interim-01.json` was explicitly incomplete (828 rows); do not present its rates as final. 400-case suite is not the set of every possible natural-language request.

## Remaining execution

### Latest handoff checkpoint (2026-09-14 14:39 KST)

Only the whole paired benchmark is still running: unified exec session28522, `.benchmarks/v70_repair/full400-paired/summary.json`. Last seen1,573/1,600: baseline perfume387/400 passed9, baseline lotion386/400 passed132; candidate perfume400/400 passed18, candidate lotion400/400 passed164. No execution errors. Wait for all1,600; do not replace final comparison with this partial count.

All candidate09 API work is complete: `all-api-09/report.json`117 calls/21 checks passed, `public-api-09/report.json`21 calls/all3830 coverage rows passed. All-regulatory09 matrix and science-parity09 passed. `installed-final-09/report.json` confirms imports from extracted final wheel and matching perfume/lotion numerical results with4 regulatory tabs.

Additional deployment closure work was necessary and is done, without deployment:

- Added `deploy/runtime_release_v70.py` and `deploy/modal_release_v70.py`.
- Assembled `tmp/modal-runtime-v70/release-01`:374 files/256,708,691 bytes; bundle SHA `f94ea18d1035fb8112f1cb21265448311eec909466aa7d4d94994325792cfb23`. Same candidate09 wheel/public-data hashes. Includes all referenced public source/history files, no raw training corpus or Windows language executable. Still private, no redistribution permission implied.
- Relocated bundle startup initially failed because `shared_runtime_v69.py` hardcoded `<profile root>/benchmarks/industrial_ingredient_registry_v1.db`, while candidate profiles correctly reference the preserved old nested asset paths. Added optional server-only `registry_path` to factory, default behavior unchanged. New V70 launcher resolves the actual hash-bound registry path from profile. `bundle-installed-check.json` now passed using installed wheel and relocated bundle, before/after seal verified. No fragrance_ai source change since candidate09.
- Existing V69 deployment entrypoint retained. New entrypoint preserves app/function names, proxy auth, CPU1/memory1024/min0/max1. NOT imported for deployment or deployed. Linux image build remains not run; prepared image will run installed preflight when deployment is explicitly authorized.
- New runtime packaging tests8 passed; combined old/new packaging suite19 passed. Duplicate-free total now **1772 passed, zero unresolved skipped/failed/error**, in `final-test-summary.json`.
- Added new regressions to release workflow; new37 CI-targeted tests passed locally with `-W error`. Ruff E4/E7/E9/F passed for changed runtime/data/test files. No remote CI/push.
- `git diff --check` has one intentional exception: exact archived concentration script's final blank line restored to original pinned hash. Do not remove it. Diff check excluding that file passes.

Documents ready: `AI_REGULATORY_DATA_V70.md`, `MODAL_DEPLOYMENT_V70_PREPARED.md`, README local-only V70 description. Final full-score report is NOT created yet; use the guarded writer after comparison completes.

Final commands after full benchmark completion:

1. `python scripts/summarize_full_scope_v70.py --input .benchmarks/v70_repair/full400-paired --output .benchmarks/v70_repair/full400-paired/final-comparison.json`
2. `python scripts/check_all_benchmark_outputs_v70.py --input .benchmarks/v70_repair/full400-paired --output .benchmarks/v70_repair/output-invariants-final.json`
3. `python scripts/write_full_scope_report_v70.py --benchmark .benchmarks/v70_repair/full400-paired/final-comparison.json --api .benchmarks/v70_repair/all-api-09/report.json --preparation .benchmarks/v70_repair/candidate-09/preparation.json --output V70_FULL_SCOPE_RESULTS.md`
4. Inspect report and final metrics. Update README with final whole400 comparison/link, verify artifacts/source/secret hygiene, final Korean handoff with no deployment claim.

Output invariant interim checks found two zero-percent diagnostic rows in baseline perfume en-10/ko-10, statusno_safe_match, no such candidate rows. Preserve original baseline artifacts. The checker now reports `candidate_checks_passed` and `all_version_checks_passed` separately, keeps all issues, and returns failure only for current-candidate defects. This does not lower the95-point threshold. Final output check still needs all1600.

### Latest continuation checkpoint (after ECHA consent, 2026-09-14 13:45 KST)

- Accepted ECHA CHEM terms only after user's explicit permission. Downloaded official lists through public UI; no CAPTCHA solved and no security warning bypass. Agent-created browser tabs were closed after acquisition.
- ECHA included-group exports: restriction 1,868, candidate 507, authorisation 140, CLP scope 10,140 rows. CLP XLSX explicitly truncated at 10,000. Visited all pagination steps to page34, verified overlapping 10,000th CAS `64742-01-4`, and captured remaining 140 CAS/EC/ECHA/index identifiers. Full names and relationships for the tail are NOT captured and remain linked to official details; do not claim all tail columns were copied.
- Tail observation `.benchmarks/v70_repair/clp_scope_tail_observed.json` was transcribed from read-only DOM, verified 140 rows / 8,173 ASCII row characters / FNV1a32 `4004645018`; subsequent archive SHA256 is recorded. `index_echa_exports_v70.py` rejects a truncated export unless this overlap and count check succeeds.
- Completed ECHA index: `.benchmarks/v70_repair/echa-official-04/eu_reach_index.json`: 43,408 total rows including history and overlapping lists, NOT 43,408 distinct chemicals. Matched 860 active materials (previous860 vs old858). Preserve per-list denominators.
- Final public bundle: `.benchmarks/v70_repair/public-evidence-registry-05/public_evidence.json`, SHA256 `495b8d1276a5bd9b77f36bf6ad3ab4f575712dd6bcccbad563bb3aa6b82a534d`, 352 source documents. Earlier registry04 output is an incomplete failed staging attempt; do not use it.
- Whole source inspection found and repaired two additional gaps: all lotion responses now attach regulatory JSON via `lotion_regulatory.py` (selected output dose takes precedence over requested dose); `rd_evidence.assess` cannot pass when published IFRA checks conflict. `change_impact` now detects regulatory-only source changes, not merely supplier/price changes.
- Current candidate09: wheel SHA256 `22767b37f437e92306423782e63cee88834b8ac91b130f504a36cb38941375bd`; profile `perfumery.v70-candidate-09.local.json`, profile SHA256 `53744fab0ef53639443f5b93d87c67c0f6d7aaca660ceacd7b4af3d2e1af9616`; catalog SHA256 `e5b023de5eeca3dc7c93225684776a79242750e130601b9572ee47e5447026aa`.
- Candidate09 whole regulatory matrix passed again: `all-regulatory-09.json`, 57,450 checks plus158 group sums. Scientific/code/data parity to candidate04 passed: `science-parity-09.json`. This does not assert identical API JSON or regulatory outcomes.
- Candidate08 whole API audit completed:117 calls21 checks passed (`all-api-08/report.json`). Actual source connection21 calls passed. Final candidate09 whole API audit RUNNING session97854 (`all-api-09`); actual source integration RUNNING session57330 (`public-api-09`). Do not alter fragrance_ai source during these runs.
- Full final suite on Python313: 1,735 reported entries, 1,725 passed, 10 skipped (including two module collection placeholders). All Modal-dependent skipped modules/tests were then run in the SDK environment:26 passed (`full-tests-sdk-supplement.xml`). PostgreSQL skip had already passed on isolated fixture. New lotion/public regressions:99 passed (`lotion-public-final-tests-02.xml`). Use `final-test-summary.json` for the duplicate-free union.
- Whole paired benchmark remains RUNNING session28522. Last observed1,312/1,600; expected completion about14:15-14:30 KST based on rate, but do not promise a timestamp. Do not call interim outputs final. `summarize_full_scope_v70.py` must succeed without `--allow-partial` before final reporting.
- Still needed: final candidate09 API/source integration receipts; finalize whole400 paired results (pass gains/losses, numerical bounds, unsupported cases), Korean results/backend handoff MD, README local-only update, final source/package/secret hygiene. No deployment or GitHub changes have been made.

1. Finish candidate-08 API and actual public data tests; address any true failures.
2. Verify candidate-04 vs candidate-08 numerical-code/data parity plus all-material cache equivalence receipt; record the coverage boundary of the quality benchmark.
3. Finish the running 1,600-result comparison, aggregate paired wins/losses, pass-rate changes, errors, and unresolved bounds/coverage. Do not infer global success from partial rows.
4. Finish full final regression/package verification; preserve all old results and record unique totals rather than summing overlapping suites.
5. Produce Korean full-scope repair and backend data/field handoff reports. State real remaining data/quality gaps and not-deployed status. Ask before production deployment.

CI source was repaired with a real JUnit minimum-test gate, separated manually authorized attestation job, and pinned PostgreSQL 17 CI service. No CI or attestation publishing has been triggered this turn.
