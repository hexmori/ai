#!/usr/bin/env python3
"""Controlled, fixed-work speculative decoding benchmark; Python 3.9+."""

import argparse
import asyncio
import aiohttp
import json
import os
import random
import time
import math
import statistics
from pathlib import Path

P = argparse.ArgumentParser(description=__doc__)
P.add_argument(
    'mode',
    choices=[
        'no_spec',
        'static_1',
        'static_3',
        'static_5',
        'static_7',
        'adaptive',
        'adaptive_fixed',
    ],
)
P.add_argument(
    '--plan',
    default='1:64:high,8:128:high,32:256:low,64:512:low,8:128:low,1:64:high',
    help='concurrency:request_count:prompt_kind, comma separated',
)
P.add_argument('--max-tokens', type=int, default=512)
P.add_argument('--warmup', type=int, default=16)
P.add_argument('--seed', type=int, default=42)
P.add_argument('--timeout', type=float, default=900)
P.add_argument('--run-id', default=os.getenv('RUN_ID', 'controlled'))
P.add_argument('--prompts', help='JSONL with kind=high or low and prompt')
P.add_argument('--natural-eos', action='store_true')

a = P.parse_args()
plan = [(int(b), int(n), k) for b, n, k in (s.split(':') for s in a.plan.split(','))]
assert a.max_tokens > 1 and a.warmup >= 0 and all(
    b > 0 and n >= b and k in ('high', 'low') for b, n, k in plan
)

base = f"http://{os.getenv('HOST', '127.0.0.1')}:{os.getenv('PORT', '30000')}"
model = os.getenv('MODEL', 'GLM-5.2')

out = Path(os.getenv('RESULT_ROOT', './results')) / a.run_id / a.mode
out.mkdir(parents=True, exist_ok=True)

corpus = {
    'high': [
        'Return only a long sequence of lines containing READY, one READY per line.',
        'Return only a long sequence of lines containing the digit 1, one per line.',
    ],
    'low': [
        'Invent a detailed travel diary about unfamiliar imaginary cultures and landscapes.',
        'Invent many distinct fictional inventions, their mechanisms, inventors and unexpected consequences.',
    ],
}

if a.prompts:
    corpus = {'high': [], 'low': []}
    for line in Path(a.prompts).read_text().splitlines():
        if line.strip():
            p = json.loads(line)
            corpus[p['kind']].append(p['prompt'])

assert all(corpus[k] for _, _, k in plan) and (not a.warmup or corpus['high'])

rng = random.Random(a.seed)


# Same prompts and request IDs across modes. Prefix IDs avoid exact full-prefix reuse.
def make(n, k, phase):
    return [
        {
            'id': f'{phase}-{i}',
            'phase': phase,
            'kind': k,
            'prompt': f'Independent task {phase}-{i}.\n' + rng.choice(corpus[k]),
        }
        for i in range(n)
    ]


warm = make(a.warmup, 'high', -1)
groups = [make(n, k, i) for i, (b, n, k) in enumerate(plan)]

(out / 'trace.jsonl').write_text(
    ''.join(json.dumps(r, ensure_ascii=False) + '\n' for g in groups for r in g)
)
(out / 'config.json').write_text(
    json.dumps(vars(a) | {'model': model, 'base': base}, indent=2)
)


def stats(rows, elapsed):
    ok = [r for r in rows if not r['error']]
    known = [r for r in ok if r['output_tokens'] is not None]
    tokens = sum(r['output_tokens'] for r in known) if len(ok) == len(known) else None

    def pct(key, p):
        v = sorted(r[key] for r in ok if r.get(key) is not None)
        return v[math.ceil(len(v) * p) - 1] if v else None

    d = dict(
        offered=len(rows),
        success=len(ok),
        failed=len(rows) - len(ok),
        usage_coverage=len(known),
        output_tokens=tokens,
        elapsed_s=elapsed,
        output_tokens_per_s=tokens / elapsed if tokens is not None and elapsed else None,
        requests_per_s=len(ok) / elapsed if elapsed else None,
    )
    for key in ('ttft_s', 'tpot_s', 'e2e_s'):
        v = [r[key] for r in ok if r.get(key) is not None]
        d['mean_' + key] = statistics.mean(v) if v else None
        for p in (50, 95, 99):
            d[f'p{p}_' + key] = pct(key, p / 100)
    return d


async def main():
    headers = (
        {'Authorization': 'Bearer ' + os.environ['API_KEY']} if os.getenv('API_KEY') else {}
    )
    async with aiohttp.ClientSession(
        headers=headers,
        timeout=aiohttp.ClientTimeout(total=a.timeout),
        connector=aiohttp.TCPConnector(limit=0),
    ) as s:
        async with s.get(base + '/health') as response:
            response.raise_for_status()

        async def req(x):
            t = time.monotonic()
            first = last = None
            n = None
            error = None
            done = False
            payload = dict(
                model=model,
                messages=[dict(role='user', content=x['prompt'])],
                temperature=0,
                max_tokens=a.max_tokens,
                stream=True,
                stream_options={'include_usage': True},
            )
            if not a.natural_eos:
                payload['ignore_eos'] = True
            try:
                async with s.post(base + '/v1/chat/completions', json=payload) as response:
                    response.raise_for_status()
                    async for raw in response.content:
                        line = raw.decode().strip()
                        if not line.startswith('data:'):
                            continue
                        data = line[5:].strip()
                        if data == '[DONE]':
                            done = True
                            break
                        obj = json.loads(data)
                        if obj.get('error'):
                            raise RuntimeError(str(obj['error']))
                        if obj.get('usage'):
                            n = obj['usage'].get('completion_tokens')
                        for c in obj.get('choices', []):
                            d = c.get('delta', {})
                            if d.get('content') or d.get('reasoning_content'):
                                last = time.monotonic()
                                first = last if first is None else first
                if not done:
                    raise RuntimeError('missing DONE')
            except Exception as e:
                error = repr(e)
            return dict(
                id=x['id'],
                phase=x['phase'],
                kind=x['kind'],
                sent=t - origin,
                end=time.monotonic() - origin,
                error=error,
                output_tokens=n,
                ttft_s=first - t if first is not None else None,
                tpot_s=(
                    (last - first) / (n - 1)
                    if first is not None and n is not None and n > 1
                    else None
                ),
                e2e_s=time.monotonic() - t,
            )

        async def group(items, b):
            it = iter(items)
            rows = []

            async def worker():
                for x in it:
                    rows.append(await req(x))

            await asyncio.gather(*(worker() for _ in range(min(b, len(items)))))
            return rows

        origin = time.monotonic()
        if warm:
            w = await group(warm, min(8, len(warm)))
            if any(r['error'] for r in w):
                raise RuntimeError(f'Warmup failed: {next(r["error"] for r in w if r["error"])}')

        origin = time.monotonic()
        stop = asyncio.Event()
        folder = out / 'metrics'
        folder.mkdir(exist_ok=True)

        async def monitor():
            while not stop.is_set():
                stamp = time.monotonic() - origin
                for endpoint, suffix in [('/metrics', 'txt'), ('/server_info', 'json')]:
                    try:
                        async with s.get(
                            base + endpoint, timeout=aiohttp.ClientTimeout(total=3)
                        ) as r:
                            if r.status == 200:
                                (
                                    folder / f'{stamp:.3f}_{endpoint[1:]}.{suffix}'
                                ).write_text(await r.text())
                    except Exception:
                        pass
                try:
                    await asyncio.wait_for(stop.wait(), 2)
                except asyncio.TimeoutError:
                    pass

        task = asyncio.create_task(monitor())
        allrows = []
        phases = []
        with (out / 'requests.jsonl').open('w') as f:
            for i, ((b, n, k), items) in enumerate(zip(plan, groups)):
                t = time.monotonic()
                print(
                    f'Phase {i}: concurrency={b} requests={n} kind={k}',
                    flush=True,
                )
                rows = await group(items, b)
                elapsed = time.monotonic() - t
                for r in rows:
                    f.write(json.dumps(r) + '\n')
                f.flush()
                allrows += rows
                d = dict(phase=i, concurrency=b, kind=k, **stats(rows, elapsed))
                phases.append(d)
                print(json.dumps(d), flush=True)
        elapsed = time.monotonic() - origin
        stop.set()
        await task
        summary = stats(allrows, elapsed)
        summary['phases'] = phases
        (out / 'summary.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))


asyncio.run(main())