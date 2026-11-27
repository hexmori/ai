#!/usr/bin/env python3
import asyncio
import aiohttp
import os
import sys
import json
import random
import time
import math
from pathlib import Path

MODE = sys.argv[1] if len(sys.argv) > 1 else ''
if MODE not in ('no_spec', 'static_1', 'static_3', 'static_5', 'static_7', 'adaptive'):
    sys.exit('Usage: python adp_dynamic.py <mode>')

BASE = f"http://{os.getenv('HOST', '127.0.0.1')}:{os.getenv('PORT', '30000')}"
MODEL = os.getenv('MODEL', 'GLM-5.2')
SEED = int(os.getenv('SEED', '42'))

OUT = Path(os.getenv('RESULT_ROOT', './results')) / os.getenv(
    'RUN_ID', time.strftime('%Y%m%d_%H%M%S')
) / MODE
OUT.mkdir(parents=True, exist_ok=True)

PHASES = [
    tuple(map(float, p.split(':')))
    for p in os.getenv(
        'PHASES', '120:0.5,120:2,120:8,120:16,120:4,120:12,180:0.5'
    ).split(',')
]
SCALE = float(os.getenv('RATE_SCALE', '1'))
CAP = int(os.getenv('MAX_INFLIGHT', '256'))
TIMEOUT = float(os.getenv('REQUEST_TIMEOUT', '600'))

assert CAP > 0 and SCALE > 0 and all(d > 0 and r > 0 for d, r in PHASES)

HEADERS = {'Authorization': 'Bearer ' + os.environ['API_KEY']} if os.getenv('API_KEY') else {}

# Synthetic natural-language tasks; replace with representative anonymized prompts via JSONL.
DEFAULT = [
    {
        'kind': 'code',
        'prompt': 'Implement a Python LRU cache with tests. Explain complexity.',
        'max_tokens': 1024,
    },
    {
        'kind': 'math',
        'prompt': 'Derive the sum of the first n squares using induction and explain each step.',
        'max_tokens': 768,
    },
    {
        'kind': 'json',
        'prompt': 'Generate a JSON array of 30 fictional inventory records with id, name, price and category. Return only JSON.',
        'max_tokens': 1024,
    },
    {
        'kind': 'qa',
        'prompt': 'Explain the difference between TCP and UDP with practical examples.',
        'max_tokens': 256,
    },
    {
        'kind': 'long',
        'prompt': 'Compare the following independent requirements and propose a distributed data processing architecture.\n'
        + '\n'.join(
            f'Requirement {i}: handle late events, retries, partitioning and monitoring for source {i}.'
            for i in range(100)
        ),
        'max_tokens': 1536,
    },
]

corpus = (
    [json.loads(l) for l in Path(os.environ['PROMPTS']).read_text().splitlines() if l.strip()]
    if os.getenv('PROMPTS')
    else DEFAULT
)
assert corpus and all(isinstance(x['prompt'], str) and int(x['max_tokens']) > 0 for x in corpus)

rng = random.Random(SEED)
trace = []
offset = 0
for phase, (duration, rate) in enumerate(PHASES):
    t = 0
    while True:
        t += rng.expovariate(rate * SCALE)
        if t >= duration:
            break
        p = rng.choice(corpus)
        trace.append(
            dict(
                id=len(trace),
                phase=phase,
                at=offset + t,
                kind=p.get('kind', 'custom'),
                prompt=p['prompt'],
                max_tokens=int(p['max_tokens']),
            )
        )
    offset += duration

(OUT / 'trace.jsonl').write_text(
    ''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in trace)
)
(OUT / 'config.json').write_text(
    json.dumps(
        dict(
            mode=MODE,
            phases=PHASES,
            rate_scale=SCALE,
            seed=SEED,
            max_inflight=CAP,
            model=MODEL,
            timeout=TIMEOUT,
        ),
        indent=2,
    )
)


async def main():
    rows = []
    pending = set()
    start = 0
    stop = asyncio.Event()

    async with aiohttp.ClientSession(
        headers=HEADERS,
        timeout=aiohttp.ClientTimeout(total=TIMEOUT),
        connector=aiohttp.TCPConnector(limit=0),
    ) as session:
        async with session.get(BASE + '/health') as resp:
            resp.raise_for_status()

        async def request(x):
            sent = time.monotonic() - start
            first = None
            usage = None
            done = False
            error = None
            try:
                payload = dict(
                    model=MODEL,
                    messages=[dict(role='user', content=x['prompt'])],
                    max_tokens=x['max_tokens'],
                    temperature=0,
                    stream=True,
                    stream_options={'include_usage': True},
                )
                async with session.post(BASE + '/v1/chat/completions', json=payload) as resp:
                    resp.raise_for_status()
                    async for raw in resp.content:
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
                            usage = obj['usage']
                        for c in obj.get('choices', []):
                            delta = c.get('delta', {})
                            if first is None and (delta.get('content') or delta.get('reasoning_content')):
                                first = time.monotonic() - start
                if not done:
                    raise RuntimeError('missing DONE')
            except Exception as e:
                error = repr(e)
            end = time.monotonic() - start
            n = usage.get('completion_tokens') if usage else None
            rows.append(
                dict(
                    id=x['id'],
                    phase=x['phase'],
                    kind=x['kind'],
                    scheduled=x['at'],
                    sent=sent,
                    end=end,
                    ttft_s=None if first is None else first - x['at'],
                    e2e_s=end - x['at'],
                    tpot_s=(
                        (end - first) / (n - 1)
                        if first is not None and n is not None and n > 1
                        else None
                    ),
                    output_tokens=n,
                    error=error,
                )
            )

        async def metrics():
            folder = OUT / 'metrics'
            folder.mkdir(exist_ok=True)
            while not stop.is_set():
                stamp = time.monotonic() - start
                try:
                    async with session.get(
                        BASE + '/metrics', timeout=aiohttp.ClientTimeout(total=3)
                    ) as r:
                        if r.status == 200:
                            (folder / f'{stamp:.3f}.txt').write_text(await r.text())
                except Exception:
                    pass
                try:
                    await asyncio.wait_for(stop.wait(), timeout=2)
                except asyncio.TimeoutError:
                    pass

        start = time.monotonic()
        monitor = asyncio.create_task(metrics())

        for x in trace:
            await asyncio.sleep(max(0, start + x['at'] - time.monotonic()))
            if len(pending) >= CAP:
                rows.append(
                    dict(
                        id=x['id'],
                        phase=x['phase'],
                        kind=x['kind'],
                        scheduled=x['at'],
                        error='client_overload_drop',
                    )
                )
                continue
            task = asyncio.create_task(request(x))
            pending.add(task)
            task.add_done_callback(pending.discard)

        await asyncio.sleep(max(0, start + offset - time.monotonic()))
        if pending:
            await asyncio.gather(*list(pending))
        elapsed = time.monotonic() - start
        stop.set()
        await monitor

    rows.sort(key=lambda x: x['id'])
    (OUT / 'requests.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in rows))

    def stats(rs):
        ok = [r for r in rs if not r['error']]
        known = [r for r in ok if r['output_tokens'] is not None]

        def pct(key, p):
            v = sorted(r[key] for r in ok if r.get(key) is not None)
            return v[max(0, math.ceil(p * len(v)) - 1)] if v else None

        return dict(
            offered=len(rs),
            success=len(ok),
            failed=len(rs) - len(ok),
            usage_coverage=len(known),
            output_tokens=(
                sum(r['output_tokens'] for r in known) if len(known) == len(ok) else None
            ),
            p50_ttft_s=pct('ttft_s', .5),
            p99_ttft_s=pct('ttft_s', .99),
            p50_tpot_s=pct('tpot_s', .5),
            p99_tpot_s=pct('tpot_s', .99),
            p99_e2e_s=pct('e2e_s', .99),
        )

    summary = stats(rows)
    summary['elapsed_s'] = elapsed
    summary['output_tokens_per_s'] = (
        summary['output_tokens'] / elapsed if summary['output_tokens'] is not None else None
    )
    summary['requests_per_s'] = summary['success'] / elapsed
    summary['phases'] = [
        dict(phase=i, **stats([r for r in rows if r['phase'] == i]))
        for i in range(len(PHASES))
    ]
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


asyncio.run(main())