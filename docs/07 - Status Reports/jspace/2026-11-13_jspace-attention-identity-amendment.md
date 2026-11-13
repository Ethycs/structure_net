# Attention transport: prospective secondary country-identity audit

Declared after discovery and before evaluation of JSPACE-ATTENTION-TRANSPORT-01.
The original protocol, data, selected layer/method, exact-answer metrics, gates,
transport fitting and classification remain frozen. This amendment adds an endpoint;
it neither replaces the original task nor repairs its score retrospectively.

Development evidence: all 12 capital sources are exact-correct; all 12 nationality
sources return the correct country name rather than its expected adjective. Across
all 24 sources, the country identity is correct. The selected layer-26 dense edit
changes country identity in 23/24, while both compact J methods change it in 0/24.
Thus exact task accuracy conflates a wording/answer-form failure with the concept
identity being studied. These are development observations, not test results.

Before opening evaluation, freeze this finite dictionary: China/Chinese/Beijing map
to China; France/French/Paris to France; Germany/German/Berlin to Germany;
Russia/Russian/Moscow to Russia. Apply only conservative whole-word normalization;
unknown or malformed answers are wrong. No LLM judge, substring search, or additions
after evaluation. Country identity deliberately ignores requested attribute.

For the same 128 held-out paired worlds, report natural source and donor identity
accuracy, jointly identity-correct pairs, each arm's donor-country success among
those pairs and all worlds, per-attribute counts, Wilson intervals, and paired exact
J-minus-matched-random comparison. No additional model forwards, altered prompts,
strength selection or model fitting. Preserve original 128-world trace and all
exact-answer metrics. Attribute-agnostic identity is not full task accuracy.

Freeze amendment/script hashes, map and creation timestamp in identity_audit_plan.json
before evaluation. Report as a secondary pilot endpoint with confirmed=false.
