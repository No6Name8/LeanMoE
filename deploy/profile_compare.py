"""Bounded A/B/C resource-profile comparison on an existing Windows VM.

Operator-only maintenance tool. No OS-cache flushing and no public session reuse.
Every outcome is checkpointed, including errors/timeouts. Eight-minute/15-minute
interpretation is explicit: the default *whole experiment* limit is 850 seconds.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time


def worker(case, store, tokens, checkpoint):
    import psutil
    import numba
    from dataclasses import asdict, replace
    from src.dashboard.resources import SafeBudget, components, load, measure
    from src.engine.runtime.auto import store_facts
    from src.engine.runtime.generate import RuntimeConfig
    from src.engine.runtime.int8_linear import current_kernel
    from src.engine.runtime.streaming import stream_generate
    from src.engine.store.chat_template import ChatTemplate
    from src.engine.store.tokenizer import load_tokenizer
    from src.engine.benchmarking import openblas_threads
    from numba.core.dispatcher import Dispatcher
    original, jit = Dispatcher.compile, []
    def compile_timed(self, *a, **kw):
        t = time.perf_counter()
        try:
            return original(self, *a, **kw)
        finally:
            jit.append(time.perf_counter()-t)
    Dispatcher.compile = compile_timed
    numba.set_num_threads(4)
    proc = psutil.Process()
    stop = threading.Event()
    samples = []
    stage = ['load']
    def sample():
        while not stop.wait(.1):
            samples.append({'stage': stage[0], 'elapsed_s': time.perf_counter()-epoch, **asdict(measure())})
    def counters():
        cpu = proc.cpu_times()
        disks = psutil.disk_io_counters(perdisk=True) or {}
        return {'cpu_s': cpu.user+cpu.system, 'process_read_bytes': proc.io_counters().read_bytes,
                'disk': {k: {'read_bytes': v.read_bytes, 'read_count': v.read_count,
                             'read_time_ms': v.read_time} for k, v in disks.items()}}
    def delta(a, b):
        return {'cpu_s': b['cpu_s']-a['cpu_s'], 'process_read_bytes': b['process_read_bytes']-a['process_read_bytes'],
                'guest_disk': {k: {n: v[n]-a['disk'].get(k, {}).get(n, 0) for n in v} for k, v in b['disk'].items()}}
    def cache(model):
        if not hasattr(model.experts, 'capacity_bytes'):
            return None
        return {'bytes': model.experts.capacity_bytes, 'slots': model.experts.num_slots,
                'occupied': len(model.experts.cached_keys())}
    rt = replace(RuntimeConfig.load(Path('configs/runtime_cache_chat.json')), max_seq=512,
                 int8_kernel='fused', memory_budget_gb=3.5, expert_cache_gb=1.2)
    tokenizer = load_tokenizer(store)
    text = 'What is 2 + 2? Give a short answer.'
    ids = tokenizer.encode(ChatTemplate.for_store(store).user_prompt(text)).ids
    mode = 'mmap' if case == 'paging' else 'cached'
    profile = 'auto' if case != 'small' else 'constrained'
    epoch = time.perf_counter()
    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    model = None
    try:
        before = counters()
        model, info = load(store, mode, rt, profile, 3.5, 1.2)
        load_s = time.perf_counter()-epoch
        loaded_counters = counters()
        start_cache = cache(model)
        checkpoint.write_text(json.dumps({'case': case, 'load_info': info, 'load_s': load_s,
                                          'cache_at_start': start_cache}), encoding='utf-8')
        live = SafeBudget(model, store_facts(store, rt, 'numpy'), components(store, rt)['transient']) if mode == 'cached' else None
        stage[0] = 'prefill'
        steps, token_ids = [], []
        last_stats = asdict(model.experts.stats)
        last_counters = counters()
        j = len(jit)
        for event in stream_generate(model, ids, tokens, 512, between_tokens=live):
            if event.kind == 'token':
                now_stats, now_counters = asdict(model.experts.stats), counters()
                steps.append({'phase': stage[0], 'seconds': event.step_seconds, 'token_id': event.token_id,
                              'source_deltas': {k: now_stats[k]-last_stats[k] for k in now_stats},
                              'cache': cache(model), **delta(last_counters, now_counters)})
                token_ids.append(event.token_id)
                last_stats, last_counters = now_stats, now_counters
                stage[0] = 'decode'
                checkpoint.write_text(json.dumps({'case': case, 'load_info': info, 'load_s': load_s,
                                                   'cache_at_start': start_cache, 'steps': steps,
                                                   'memory_samples': samples}), encoding='utf-8')
            else:
                done = event
        # KV was released by stream_generate; source cache remains only inside
        # this private worker until finally. It is never reused by another visitor.
        return {'case': case, 'mode': mode, 'profile': info['profile'], 'load_info': info,
                'backend': model.backend.name, 'kernel': current_kernel(),
                'threads': {'openblas': openblas_threads(), 'numba': numba.get_num_threads()},
                'prompt_ids': ids, 'token_ids': token_ids, 'requested_tokens': tokens,
                'load_s': load_s, 'load_io': delta(before, loaded_counters),
                'prefill_s': done.time_to_first_token_s, 'decode_tokens_s': done.decode_tokens_per_s,
                'total_s': done.total_seconds, 'stop': done.stop_reason, 'error': done.error,
                'jit_dispatch_s': sum(jit[j:]), 'cache_at_start': start_cache, 'cache_at_end': cache(model),
                'steps': steps, 'memory_samples': samples,
                'budget_events': live.events if live else []}
    finally:
        stop.set()
        sampler.join()
        if model is not None:
            model.experts.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--store', required=True, type=Path)
    ap.add_argument('--manifest', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--child', choices=('small', 'automatic', 'paging'))
    ap.add_argument('--tokens', type=int, default=16)
    ap.add_argument('--seconds', type=int, default=850)
    a = ap.parse_args()
    if a.child:
        print(json.dumps(worker(a.child, a.store, a.tokens, a.out)))
        return
    env = {**os.environ, 'OPENBLAS_NUM_THREADS': '4', 'NUMBA_NUM_THREADS': '4'}
    import psutil
    from src.dashboard.system import cpu_name
    result = {'purpose': 'same-machine A/B/C configuration diagnostic, unequal RAM consumption reported explicitly',
              'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'manifest': json.loads(a.manifest.read_text(encoding='utf-8-sig')),
              'manifest_sha256': hashlib.sha256(a.manifest.read_bytes()).hexdigest(),
              'source_sha256': {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
                                ['src/dashboard/resources.py', 'deploy/profile_compare.py', 'src/engine/runtime/generate.py',
                                 'src/engine/runtime/streaming.py', 'src/engine/runtime/int8_linear.py']},
              'os_cache': 'uncontrolled, no flush, preceding cases and earlier reads may warm it',
              'process_layout': 'fresh process/model/KV for every case; no shared session state',
              'machine': {'cpu': cpu_name(), 'logical_cores': psutil.cpu_count(),
                          'physical_ram_bytes': psutil.virtual_memory().total, 'execution_location': 'server'},
              'prompt': 'What is 2 + 2? Give a short answer.', 'max_seq': 512,
              'sampling_s': .1, 'disk_scope': 'Windows guest disk counters across all processes, not host physical media',
              'timing': 'decode excludes load/prefill; forward+argmax only; byte counters can straddle asynchronous prefetch boundaries',
              'runs': []}
    schedule = [(0, c, 4) for c in ('small', 'automatic', 'paging')]
    schedule += [(r, c, a.tokens) for r, order in [(1, ('automatic','paging','small')), (2, ('paging','small','automatic'))] for c in order]
    deadline = time.monotonic()+a.seconds
    a.out.parent.mkdir(parents=True, exist_ok=True)
    for round_id, case, tokens in schedule:
        remaining = deadline-time.monotonic()
        if remaining < 15:
            result['stopped'] = 'whole-experiment deadline; remaining cases not attempted'
            break
        t = time.monotonic()
        partial = a.out.with_suffix('.worker.json')
        if partial.exists():
            partial.unlink()
        try:
            p = subprocess.Popen([sys.executable, '-m', 'deploy.profile_compare', '--child', case, '--tokens', str(tokens),
                                  '--store', str(a.store), '--manifest', str(a.manifest), '--out', str(partial)],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
            stdout, stderr = p.communicate(timeout=min(200, remaining))
            row = json.loads(stdout) if p.returncode == 0 else {'case': case, 'error': 'worker failed', 'exit_code': p.returncode}
            if p.returncode:
                # Engine/path diagnostics are operator-private, not public artifacts.
                with a.out.with_suffix('.private.log').open('a', encoding='utf-8') as log:
                    log.write(stderr)
        except subprocess.TimeoutExpired:
            for process in [*psutil.Process(p.pid).children(recursive=True), psutil.Process(p.pid)]:
                try:
                    process.kill()
                except psutil.Error:
                    pass
            p.communicate()
            row = json.loads(partial.read_text()) if partial.exists() else {'case': case}
            row.update(error='worker timeout; process tree terminated', requested_tokens=tokens)
        row.update(round=round_id, child_wall_s=time.monotonic()-t)
        result['runs'].append(row)
        result['elapsed_s'] = a.seconds-(deadline-time.monotonic())
        a.out.write_text(json.dumps(result, separators=(',', ':'))+'\n', encoding='utf-8')
    result['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    a.out.write_text(json.dumps(result, separators=(',', ':'))+'\n', encoding='utf-8')
    print('A/B/C outcomes preserved:', len(result['runs']))


if __name__ == '__main__':
    main()
