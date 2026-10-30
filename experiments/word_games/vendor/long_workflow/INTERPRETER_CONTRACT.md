# Museum-to-garden translation contract
You are a fresh interpreter of one event in a continuing external-memory task. Translate museum exhibit stewardship into garden seed stewardship and propose the minimal typed source-state operations that implement the source request. Preserve identity, dates, custody, approvals, slot occupancy, scope and rejection semantics. Provide meaningful garden-language prose, not merely field renaming. Source galleries map to garden plots (gal:north→plot:north, gal:south→plot:south, gal:archive→plot:archive); source items loan:X map to seed:X; display slots become planting beds; reservations become plantings with a return deadline. Metal/paper/stone/textile are compatibility classes (retain these labels), restricted items require a narrow stewardship approval, public/research departments are garden districts, and one bed cannot host two simultaneous plantings. A material exception is narrow to its approval, not a global compatibility rule change. Permission to propose does not grant approval.

Read external memory selectively using ONLY these commands, choosing queries yourself:
- `python retrieve.py context`: short host-bound context, revision, ancestor versions, referents; authorized entity IDs stay host-side.
- `python retrieve.py read ID...`: exact scoped records.
- `python retrieve.py search WORDS...`: lexical candidate search, not final identity resolution.
- `python retrieve.py occupancy GALLERY_ID [SLOT]`: indexed current occupied/empty beds.
- `python retrieve.py query '{"event_id":"EVENT_ID","order":"creation_desc","limit":20}'`: provenance-indexed current loan records. Optional gallery_id, after_sequence supported.
- `python retrieve.py history EVENT_ID`: external past-event source request, actual proposal and outcome.
- `python retrieve.py history-search WORDS...`: external past-event candidate IDs/snippets containing all supplied words, case-insensitive.
History contains only completed events of this project and is external memory, not conversation. It records failed/clarification outcomes as well as successful actions; never treat a failed proposal as committed. A past request may identify an object through creation provenance or the actual operation. Context referents are explicitly tracked discourse bindings; do not infer a different 'it'. Approval identifiers present in an event are exact keys. All dates use YYYY-MM-DD; through/until mean inclusive return-by date. Do not read session.json, history.json, backend source, sibling directories, oracle, future events, prior prompts, or state dumps; do not mutate memory. This is a methodological restriction, not an OS sandbox. The trusted host grants a fixed curator role with all three galleries, all operation types, and a static accession ID roster for the entire project; this permission does not supply stewardship approval.

Submit exactly one JSON response in response.json:
{"target_text":"garden-language translation of this request, including conditional/rejected effect where relevant","proposal":{...}}
The host executes your proposal once after this response. Do not claim host-confirmed success: describe requested or anticipated outcome. An explicit request to attempt an invalid operation should still produce that attempted transaction, not an invented correction. A genuinely ambiguous entity should produce clarification, never a guess. A bare title matching multiple entities requires clarification even if one was mentioned recently; an explicit pronoun, provenance qualifier, or unique exact ID may disambiguate.

Transaction envelope: {"kind":"transaction","context_id":CONTEXT_ID,"tx_id":EVENT_ID,"expected_revision":CONTEXT_REVISION,"rule_versions":CONTEXT_RULE_VERSIONS,"operations":[OP,...]}.
Exact typed operations (all listed fields mandatory, no extra fields):
- reserve: {"op":"reserve","loan_id":ID,"gallery_id":ID,"slot":SLOT,"due":DATE,"approval_id":ID_OR_NULL}
- transfer: {"op":"transfer","loan_id":ID,"to_gallery_id":ID,"slot":SLOT,"due":DATE,"approval_id":ID_OR_NULL}; works for correcting an existing reservation in its same gallery/slot too.
- cancel: {"op":"cancel","loan_id":ID}; removes planting, returns custody home, does not plant in a home bed.
- annotate: {"op":"annotate","loan_id":ID,"text":EXACT_NOTE}
- create: {"op":"create","loan_id":NEW_ID,"title":TITLE,"home":GALLERY_ID,"medium":"metal|paper|stone|textile","restricted":BOOLEAN}
Clarification envelope: {"kind":"clarify","question":QUESTION}.

All operations in one transaction are atomic; final occupancy is validated, so an atomic swap is possible. Any rejection leaves source state unchanged. Contexts contain current revision and all scoped ancestor rule versions. A stale context cannot commit, even if the proposed action would satisfy the new rule. An identical transaction replay returns its old receipt without applying twice; a changed proposal with the same transaction ID conflicts. Event-specific instructions control intentional stale submissions and exact retries. Creation sequence and creation event identify newly created versus older same-title objects. Never rewrite full state or silently substitute a different action/date/entity.
