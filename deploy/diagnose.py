"""Bounded operator diagnostic, not a speedup benchmark. Does not flush OS caches.

Run only while the public engine is idle and reserved for maintenance. Each mode
gets a fresh child process and two identical four-token requests in that process.
The second request has a warm engine; OS file-cache state remains uncontrolled.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def child(mode, store, threads):
    import psutil
    start = time.perf_counter()
    import numba
    from numba.core.dispatcher import Dispatcher
    compiled = []
    original = Dispatcher.compile
    def compile_timed(self, *args, **kwargs):
        t = time.perf_counter()
        try:
            return original(self, *args, **kwargs)
        finally:
            compiled.append(time.perf_counter() - t)
    Dispatcher.compile = compile_timed
    numba.set_num_threads(threads)
    from dataclasses import replace
    from src.engine.runtime.generate import RuntimeConfig, load_model
    from src.engine.runtime.streaming import stream_generate
    from src.engine.store.tokenizer import load_tokenizer
    from src.engine.store.chat_template import ChatTemplate
    from src.engine.benchmarking import openblas_threads
    imports = time.perf_counter() - start
    proc = psutil.Process()
    def counters():
        cpu = proc.cpu_times()
        disks = psutil.disk_io_counters(perdisk=True) or {}
        return {'cpu_s': cpu.user + cpu.system, 'process_read_bytes': proc.io_counters().read_bytes,
                'disks': {k: {'read_bytes': v.read_bytes, 'read_count': v.read_count,
                              'read_time_ms': v.read_time} for k, v in disks.items()}}
    def delta(a, b, wall):
        return {'wall_s': wall, 'process_cpu_s': b['cpu_s'] - a['cpu_s'],
                'cpu_percent_of_all_vcpus': (b['cpu_s'] - a['cpu_s']) / wall / psutil.cpu_count() * 100,
                'process_read_bytes': b['process_read_bytes'] - a['process_read_bytes'],
                'system_disk_deltas': {k: {n: v[n] - a['disks'].get(k, {}).get(n, 0) for n in v}
                                       for k, v in b['disks'].items()}}
    rt = replace(RuntimeConfig.load(Path('configs/runtime_cache_chat.json')), max_seq=512,
                 memory_budget_gb=3.5, expert_cache_gb=1.2, int8_kernel='fused')
    tokenizer = load_tokenizer(store)
    prompt = 'What is 2 + 2? Give a short answer.'
    ids = tokenizer.encode(ChatTemplate.for_store(store).user_prompt(prompt)).ids
    before, t = counters(), time.perf_counter()
    model, _ = load_model(store, mode, 512, 3.5, None, rt)
    loaded = delta(before, counters(), time.perf_counter() - t)
    runs = []
    try:
        for index in range(2):
            before, t, j = counters(), time.perf_counter(), len(compiled)
            steps, tokens = [], []
            last_counters = before
            for event in stream_generate(model, ids, max_new_tokens=4, max_seq=512):
                if event.kind == 'token':
                    current = counters()
                    steps.append({'seconds': event.step_seconds, 'stats': event.step_stats,
                                  'measurements': delta(last_counters, current, event.step_seconds)})
                    last_counters = current
                    tokens.append(event.token_id)
                else:
                    if event.error:
                        raise RuntimeError('Engine diagnostic failed')
                    done = event
            runs.append({'engine_state': 'fresh model' if index == 0 else 'warm engine, retained expert cache',
                         'os_file_cache': 'uncontrolled; no flush; prior reads and manifest verification can warm it',
                         'prefill_s': done.time_to_first_token_s, 'decode_tokens_s': done.decode_tokens_per_s,
                         'token_ids': tokens, 'steps': steps, 'stop_reason': done.stop_reason,
                         'jit_dispatch_seconds': sum(compiled[j:]),
                         **delta(before, counters(), time.perf_counter() - t)})
    finally:
        model.experts.close()
    return {'mode': mode, 'fresh_process': True, 'import_seconds': imports, 'load': loaded,
            'threads': {'openblas': openblas_threads(), 'numba': numba.get_num_threads()},
            'runs': runs}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--store', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--threads', type=int, choices=(1, 2, 4), default=4)
    ap.add_argument('--child', choices=('cached', 'mmap'))
    a = ap.parse_args()
    if a.child:
        print(json.dumps(child(a.child, a.store, a.threads)))
        return
    deadline = time.monotonic() + 850
    result = {'purpose': 'bounded cloud diagnosis; not a benchmark or speedup claim',
              'prompt': 'What is 2 + 2? Give a short answer.', 'max_new_tokens': 4,
              'max_seq': 512, 'budget_gb': 3.5, 'starting_cache_gb': 1.2,
              'execution_location': 'server', 'store_drive': a.store.drive,
              'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'jit_definition': 'time inside Numba Dispatcher.compile, including cached specialization loading; overlaps phase wall time',
              'disk_definition': 'system-wide Windows disk counters, including unrelated processes; not isolated device throughput',
              'runs': []}
    env = {**os.environ, 'OPENBLAS_NUM_THREADS': str(a.threads), 'NUMBA_NUM_THREADS': str(a.threads)}
    try:
        for mode in ('cached', 'mmap'):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired('diagnostic', 850)
            p = subprocess.run([sys.executable, '-m', 'deploy.diagnose', '--child', mode,
                                '--store', str(a.store), '--out', str(a.out), '--threads', str(a.threads)],
                               capture_output=True, text=True, timeout=remaining, env=env)
            if p.returncode:
                result['error'] = 'child failed; private operator diagnostics required'
                break
            result['runs'].append(json.loads(p.stdout))
    except subprocess.TimeoutExpired:
        result['error'] = '850-second bound reached; child terminated'
    finally:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print('Diagnostic saved; modes completed:', len(result['runs']))


if __name__ == '__main__':
    main()
