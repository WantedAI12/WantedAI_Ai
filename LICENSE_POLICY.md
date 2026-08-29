# Distribution and data-license policy

`perfumery-ai-core` is distributed under the proprietary terms in
[`LICENSE`](LICENSE). It is **not** MIT-licensed. The `license` metadata in
`pyproject.toml` intentionally points at that file so a package consumer does
not receive a conflicting SPDX claim.

## Wheel contents

The source checkout contains `fragrance_ai/data/reference_fragrances.db` only
as an unverified, workspace-local historical compatibility input. Its upstream
license has not been established. Setuptools excludes it from every wheel.
`HistoricalReferenceCorpus` supports an empty corpus when the file is absent;
this is expected in an installed package and is not an error.

All other bundled data remain subject to their individual provenance and
claim-boundary entries in `fragrance_ai/data/data_manifest.json`. Inclusion in
a wheel is not a grant of rights beyond the proprietary software license.

## Commercial dependencies and release boundary

The `commercial` extra provides `cryptography` for signed operator-evidence
integrations and `PyJWT[crypto]` for issuer/audience/JWKS-verified OIDC access.
Installing it does not approve a formula. Qualified and
commercial requests remain fail-closed until supplier IFRA certificate, SDS,
lot COA, quantitative allergen statement, and required external sign-off are
verified by the application's release workflow.

The embedded local IFRA screen is a deliberately incomplete Amendment 50
demonstration subset. It is suitable only as a prototype screen; absence of a
rule is unknown, never unrestricted or commercially compliant.
