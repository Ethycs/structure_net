# XSum text rendering: padding versus end-of-sequence

Exploratory diagnostic `ncm-xsum-render-parity-v1`. The frozen four-arm semantic
probe deliberately used `ncm_llama.decode_ids`: it stops at either EOS or PAD.
The released `scripts/eval_shard.py::dec` instead stops only at EOS and removes
PAD tokens. This difference was found after the native arm completed128 valid
judgments with3 passes while the other arms were still running. A read-only
comparison found changed text in native16, initial18, MSE22 and distill17 cases.
No new outcome is being called prospective confirmation.

Verify the existing `ncm_decode_policy.render_author` against the exact `dec`
function extracted from the pinned author script for all512 saved raw word
sequences. Use the same local GPT2 tokenizer and published EOS/PAD identities;
strip only outer whitespace after either decoder. Bind original source/text
packets and raw output bytes. Do not regenerate tokens, train, choose favorable
examples, edit earlier results or change the running grader.

Save corrected texts and exact anonymous packets for all four arms. Record the
changed case IDs, old/new text, token counts and packet identities. Replay all
128 completed native-arm raw judgments before deriving a conservative bound:
unchanged original passes plus every changed case. This is an upper bound on
the score attainable by the rendering change alone when unchanged first
judgments are retained; it is not measured corrected semantic accuracy.

This CPU-only diagnostic uses locked Pixi ncm and fresh root
`20261027_xsum_render_parity_v1`, makes zero judge requests and leaves final320
sealed. No source or reference targets enter training. The main curriculum
continues unchanged. All completed probe judgments remain first decisions for
their exact packets. Any subsequent corrected semantic evaluation requires a
new, explicit protocol and the full completed probe cache; unchanged packets
must reuse their existing firsts. The current four-arm study remains reported
under its actual EOS-or-PAD rendering policy.
