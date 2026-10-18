#!/usr/bin/env python3
"""Requests that share a long prefix and start together: time to the first token, cached tokens, and the reply tokens.

  burst_reuse.py make --workload burst --lines 2800 --streams 4 --output-tokens 128 --temperature 1 --top-k 20 \
      --top-p .95 --thinking-on max --output burst.json
  burst_reuse.py run --fixtures burst.json --base-url http://127.0.0.1:8020 --concurrency 4 --output release.jsonl
  burst_reuse.py compare release.jsonl modified.jsonl

A vLLM server does not accept the text priority field and does not give the reply token ids in the same form:
use 'make --no-priority' and 'run --no-token-ids' for it. 'make --tag X' puts X into the first line of the prompt, so a
server that got the prompt before sees a new prefix.

'make' and 'compare' send nothing; only 'run' sends requests. The requests are frozen JSON bodies with a seed, so two
servers get the same bytes. The prompt size is approximate: read the actual size from usage.prompt_tokens in the
output. 'compare' needs the reply token ids, which TensorFold gives with return_token_ids. An output file is never
replaced.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import urllib.request

def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def output(path):
    p = Path(path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    return p.open('x')


def make(args):
    text = ''.join(f'line {n}: alpha beta gamma delta epsilon; preserve input order.\n' for n in range(args.lines))
    def body(content, tokens=None):
        return dict(model=args.model, messages=[dict(role='user', content=content)],
                    max_tokens=tokens if tokens is not None else args.output_tokens,
                    temperature=args.temperature, top_k=args.top_k, top_p=args.top_p, min_p=0,
                    seed=777, ignore_eos=True, return_token_ids=True,
                    **({} if args.no_priority else dict(priority='normal')),
                    chat_template_kwargs=dict(enable_thinking=bool(args.thinking_on),
                                              reasoning_effort=args.thinking_on or 'high'))
    head = 'Burst fixture' + (f' {args.tag}' if args.tag else '') + '\n'
    if args.workload == 'burst':
        rows = [dict(name=f'sibling-{i}', delay_s=0, body=body(head + text + f'\nQuestion {i}: summarize.'))
                for i in range(args.streams)]
    elif args.workload == 'chat':
        base = body('Chat fixture\n' + text)
        new = dict(base, messages=base['messages'] + [dict(role='user', content='Now list three words.')])
        rows = [dict(name=name, delay_s=0, body=b) for name, b in (('cold', base), ('identical', base), ('new-user', new))]
    elif args.workload == 'mixed':
        rows = [dict(name=f'decode-{i}', delay_s=0, body=body(f'Count integers from {i} without stopping.', 2000))
                for i in range(4)]
        rows += [dict(name=f'fill-{i}', delay_s=1, body=body('Mixed fixture\n' + text + f'\nQuestion {i}'))
                 for i in range(4)]
    else:
        rows = [dict(name='cold', delay_s=0, body=body('Cold fixture\n' + text))]
    doc = dict(workload=args.workload, requests=rows, approximate=True)
    with output(args.output) as f:
        json.dump(doc, f, indent=2)
        f.write('\n')
    print(f'Wrote {len(rows)} frozen requests; actual token counts must be read from server usage.')


def run(args):
    doc = json.loads(Path(args.fixtures).read_text())
    if doc['workload'] == 'chat' and args.concurrency != 1:
        raise ValueError('Chat replay requires --concurrency 1')
    origin = time.perf_counter()
    def one(row):
        wait = origin + row['delay_s'] - time.perf_counter()
        if wait > 0:
            time.sleep(wait)
        body = dict(row['body'], stream=True, stream_options=dict(include_usage=True))
        result = dict(name=row['name'], request_sha256=digest(row['body']), body=row['body'])
        began = time.perf_counter()
        first, stats, usage, pieces = None, None, None, []
        try:
            req = urllib.request.Request(args.base_url.rstrip('/') + '/v1/chat/completions',
                                         json.dumps(body).encode(), {'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=900) as response:
                for raw in response:
                    line = raw.decode().strip()
                    if not line.startswith('data:') or line[5:].strip() == '[DONE]':
                        continue
                    event = json.loads(line[5:])
                    if event.get('error'):
                        raise RuntimeError(str(event['error']))
                    delta = (event.get('choices') or [{}])[0].get('delta') or {}
                    text = ''.join(delta.get(k) or '' for k in ('content', 'reasoning_content', 'reasoning'))
                    if text:
                        first = first if first is not None else time.perf_counter() - began
                        pieces.append(text)
                    stats = event.get('tensorfold') or stats
                    usage = event.get('usage') or usage
            ids = (stats or {}).get('token_ids')
            if args.no_token_ids:
                ids = ids if isinstance(ids, list) and ids else None
            elif not isinstance(ids, list) or not ids:
                raise RuntimeError('Missing return_token_ids; exactness cannot be checked')
            result.update(status='ok', ttft_s=first, elapsed_s=time.perf_counter() - began,
                          token_ids=ids, token_sha256=digest(ids) if ids else None, stats=stats, usage=usage,
                          text=''.join(pieces))
        except Exception as exc:
            result.update(status='error', error=str(exc), elapsed_s=time.perf_counter() - began)
        return result
    with output(args.output) as f, ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        rows = list(pool.map(one, doc['requests']))
        for row in rows:
            f.write(json.dumps(row) + '\n')
    print(json.dumps([dict(name=r['name'], status=r['status'], ttft_s=r.get('ttft_s'),
                           cached=(r.get('stats') or {}).get('cached'),
                           prompt_tokens=(r.get('usage') or {}).get('prompt_tokens')) for r in rows]))
    if any(r['status'] != 'ok' for r in rows):
        raise SystemExit(1)


def compare(args):
    left, right = [[json.loads(line) for line in Path(p).read_text().splitlines()] for p in (args.left, args.right)]
    if len(left) != len(right) or not left:
        raise SystemExit('Result count mismatch/empty')
    for a, b in zip(left, right):
        if (a['name'] != b['name'] or a.get('status') != 'ok' or b.get('status') != 'ok'
                or a['request_sha256'] != b['request_sha256'] or a['token_ids'] != b['token_ids']):
            raise SystemExit(f"Exactness failed for {a['name']}")
    print(f'Exact token equality: {len(left)} requests')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    m = sub.add_parser('make')
    m.add_argument('--workload', choices=['burst', 'chat', 'mixed', 'cold'], required=True)
    m.add_argument('--lines', type=int, default=2800)
    m.add_argument('--streams', type=int, default=4)
    m.add_argument('--output-tokens', type=int, default=128)
    m.add_argument('--thinking-on', choices=['low', 'medium', 'high', 'max'])
    m.add_argument('--model', default='glm-5.3-flash')
    m.add_argument('--temperature', type=float, default=1)
    m.add_argument('--top-k', type=int, default=20)
    m.add_argument('--top-p', type=float, default=0.95)
    m.add_argument('--tag', default='', help='text for the first line of the prompt: a new prefix for a warm server')
    m.add_argument('--no-priority', action='store_true', help='no priority field (a vLLM server does not accept the text)')
    m.add_argument('--output', required=True)
    r = sub.add_parser('run')
    r.add_argument('--base-url', default='http://127.0.0.1:8021')
    r.add_argument('--fixtures', required=True)
    r.add_argument('--output', required=True)
    r.add_argument('--concurrency', type=int, default=4)
    r.add_argument('--no-token-ids', action='store_true', help='record the times when the server gives no token ids')
    c = sub.add_parser('compare')
    c.add_argument('left')
    c.add_argument('right')
    args = p.parse_args()
    {'make': make, 'run': run, 'compare': compare}[args.command](args)


if __name__ == '__main__':
    main()
