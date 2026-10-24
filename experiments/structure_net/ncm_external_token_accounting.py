"""Estimate recorded external NCM API usage without reading sealed source files."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys

DEST = Path('data/experiments/ncm_paper_regime/20261024_external_token_audit_v2')


def prompt_tokens(request, encoding):
    messages = request.get('messages', [])
    total = 3
    for message in messages:
        total += 4
        for name, value in message.items():
            text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
            total += len(encoding.encode(text, disallowed_special=()))
    return total


def output_tokens(response, encoding):
    count = 0
    for choice in response.get('choices', []):
        message = choice.get('message', {})
        for key in ('content', 'reasoning_content', 'tool_calls'):
            value = message.get(key)
            if value:
                text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
                count += len(encoding.encode(text, disallowed_special=()))
    return count


def main():
    if DEST.exists():
        raise ValueError('fresh append-only external usage inventory required')
    roots = sorted(p for p in Path('data/experiments').iterdir() if p.is_dir() and p.name.startswith('ncm'))
    command = ['rg', '--files', '-g', '*.json', '-g', '!**/producer_source/**', *map(str, roots)]
    names = subprocess.run(command, check=True, capture_output=True, text=True).stdout.splitlines()
    paths = []
    for name in names:
        path = Path(name)
        # Filter by FILE NAME before opening anything; never search source files for usage.
        if ('20261024_clean_exchange_partitions_v2' in path.parts
                or path.name in ('final_short_sources.json', 'final_long_sources.json')):
            continue
        if (set(path.parts) & {'attempts', 'requests', 'raw_requests', 'calls'}
                or path.name.endswith(('.response.json', '_response.json'))
                or ('raw' in path.parts and any(v in path.name for v in ('response', 'request', 'call')))):
            paths.append(path)
    sys.path.append(str(Path('data/experiments/ncm_memory/dependencies').resolve()))
    import tiktoken
    encoding = tiktoken.get_encoding('cl100k_base')
    counts, by_model, events, seen = Counter(), defaultdict(Counter), [], {}
    seen_provider_ids = set()
    for path in sorted(paths):
        if path.stat().st_size > 64 * 1024**2:
            counts['oversized_raw_files_not_parsed'] += 1
            continue
        try:
            value = json.loads(path.read_text())
        except (ValueError, OSError):
            counts['unreadable_or_partial_JSON_files'] += 1
            continue
        if not isinstance(value, dict):
            continue
        request = value.get('request', value.get('payload'))
        if not isinstance(request, dict) or not isinstance(request.get('messages'), list):
            # A canary's request and response are separate; recover its exact request.
            sibling = path.with_name(path.name.replace('.response.json', '.request.json'))
            if path.name.endswith('.response.json') and sibling.exists():
                request = json.loads(sibling.read_text())
            else:
                continue
        response = value.get('response', {})
        if not isinstance(response, dict):
            response = {}
        model = request.get('model', response.get('model', 'unknown'))
        signature = {k: value.get(k) for k in ('seconds', 'elapsed_seconds', 'error_type', 'error')}
        signature.update(request=request, response=response, http_status=value.get('http_status'),
            response_text_if_unparsed=value.get('response_text') if not response else None)
        digest = hashlib.sha256(json.dumps(signature, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        provider_id = response.get('id')
        # Claude/Qwen IDs identify a call; the SDK's constant ID does not.
        identity = (model, provider_id) if provider_id and provider_id not in ('codex-sdk', 'codex_sdk') else None
        if digest in seen or (identity is not None and identity in seen_provider_ids):
            counts['duplicate_saved_copies_excluded'] += 1
            continue
        seen[digest] = str(path)
        if identity is not None:
            seen_provider_ids.add(identity)
        counts['unique_recorded_attempts'] += 1
        row = {'path': str(path), 'event_sha256': digest, 'model': model}
        usage = response.get('usage', {}) or {}
        prompt = usage.get('prompt_tokens', usage.get('input_tokens', 0)) or 0
        completion = usage.get('completion_tokens', usage.get('output_tokens', 0)) or 0
        total = max(usage.get('total_tokens', 0) or 0, prompt + completion)
        visible_input, visible_output = prompt_tokens(request, encoding), output_tokens(response, encoding)
        text = ' '.join(str(c.get('message', {}).get('content', '')) for c in response.get('choices', [])).lower()
        rejection = any(term in text for term in ('weekly limit', 'credit balance is too low', 'usage limit reached'))
        row.update(provider_reported_input_tokens=prompt, provider_reported_output_tokens=completion,
            provider_reported_total_tokens=total, visible_input_tokens=visible_input, visible_output_tokens=visible_output)
        if total > 0:
            # Claude Code often reports only 3 input tokens despite a long prompt.
            # Preserve provider counters, but include the observed prompt in the estimate.
            estimated_input = max(prompt, visible_input)
            estimated_output = max(completion, visible_output)
            row.update(category='provider_usage_with_visible_input_floor', input_tokens=estimated_input,
                output_tokens=estimated_output, total_tokens=max(total, estimated_input + estimated_output))
            counts['attempts_with_nonzero_provider_usage'] += 1
            counts['provider_reported_input_tokens'] += prompt
            counts['provider_reported_output_tokens'] += completion
            counts['provider_reported_total_tokens'] += total
            counts['provider_input_reports_below_visible_prompt'] += int(prompt < visible_input)
            counts['provider_output_reports_below_visible_response'] += int(completion < visible_output)
        elif rejection or (value.get('http_status') not in (None, 200)):
            row.update(category='rejected_or_quota_no_measured_usage', input_tokens=0, output_tokens=0, total_tokens=0)
            counts['rejected_or_quota_attempts_excluded_from_estimate'] += 1
        elif response.get('choices'):
            row.update(category='visible_text_estimate_usage_missing_or_zero', input_tokens=visible_input,
                output_tokens=visible_output, total_tokens=visible_input + visible_output)
            counts['response_attempts_with_missing_or_zero_usage'] += 1
            counts['estimated_visible_input_tokens_for_missing_usage'] += visible_input
            counts['estimated_visible_output_tokens_for_missing_usage'] += visible_output
        else:
            row.update(category='transport_or_unparsed_response_usage_unknown', input_tokens=0, output_tokens=0, total_tokens=0,
                attempted_visible_input_tokens=visible_input)
            counts['transport_or_unparsed_attempts_usage_unknown'] += 1
            counts['attempted_input_tokens_for_unknown_usage'] += visible_input
        counts['hybrid_estimate_input_tokens'] += row['input_tokens']
        counts['hybrid_estimate_output_tokens'] += row['output_tokens']
        counts['hybrid_estimate_total_tokens'] += row['total_tokens']
        if row['category'] not in ('rejected_or_quota_no_measured_usage', 'transport_or_unparsed_response_usage_unknown'):
            counts['observed_successful_response_visible_input_tokens'] += visible_input
            counts['observed_successful_response_visible_output_tokens'] += visible_output
        counts['all_recorded_attempts_visible_input_tokens'] += visible_input
        counts['all_recorded_responses_visible_output_tokens'] += visible_output
        by_model[model]['attempts'] += 1
        for key in ('input_tokens', 'output_tokens', 'total_tokens'):
            by_model[model][key] += row[key]
        by_model[model][row['category']] += 1
        events.append(row)
        if len(events) % 1000 == 0:
            print(json.dumps({'unique_accounted_external_attempts': len(events)}), flush=True)
    DEST.mkdir()
    (DEST / 'events.json').write_text(json.dumps(events, indent=2))
    result = {'scope': 'saved external teacher/judge/other API calls under all local ncm experiment roots',
        'roots': [str(p) for p in roots], 'candidate_raw_files': len(paths), 'counts': dict(counts),
        'by_model': {k: dict(v) for k, v in by_model.items()},
        'estimate_rule': 'max(provider report, cl100k visible text) separately for input and output of observed replies; provider total retained if larger',
        'reason_for_v2': 'v1 trusted nonzero usage with severe Claude Code prompt underreporting; v2 uses observed prompt floor and deduplicates nonconstant provider IDs',
        'excluded_from_estimate': 'known quota/rejections; transport/unparsed failures with unknown consumption',
        'not_included': 'agent-run token meter; unreturned internal wrapper context/reasoning; external calls without saved raw request records',
        'duplicate_rule': 'canonical request/response/timing/error event digest, not constant provider response IDs',
        'sealed_final_source_files_read': False, 'local_SGD_tokens_are_not_external_API_tokens': True,
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'events_sha256': hashlib.sha256((DEST / 'events.json').read_bytes()).hexdigest()}
    (DEST / 'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
