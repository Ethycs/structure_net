# Llama 3B shared summarization prompt correction

Date: 2026-10-13. Parent `llama32-3b-summary-throughput-v1`.
The original development screen has been inspected. No held-out model outputs
have been generated. Keep the first prompt's results and classifications.

Both optimized FP16 and existing Q4 omitted all final decisions on the roughly
2,000-token single-fragment development case. They instead summarized repetitive
background discussion. Q4 draft 16 also introduced an extra fragment. The original
screen therefore cannot establish the required quality. This is not a reason to
drop the failed input family or claim parity is sufficient.

Change the common task wrapper, identically for all arms: use a short system
instruction and place the full output requirements after the source fragments.
Preserve every source fragment, fact, seed, and generation cap. New tokenized
fixtures must remain within the 6,300-token raw prompt cap. If formatting exceeds
the cap, stop fixture preparation; do not silently truncate source evidence.

Create `*_prompt_v2.json` fixtures and new runtime roots. Run the four development
families again before any held-out evaluation. Select draft length 8 provisionally:
in the original Q4 screen it retained the scalar output lengths, had lower total
latency than 16, and delivered essentially the same token throughput. Final
selection remains on development data. The first prompt's measurements do not
become passing evidence after this correction.
