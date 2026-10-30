"""Measured dashboard allocation policy. GB is decimal; no OS/cache RAM cap.

Physical reserve: max(1 GB, 12% of physical RAM), plus 256 MB race/measurement
slack. Future KV/scratch/activation/JIT allocations are reserved separately.
Available RAM already excludes this process's resident pages. Private commit is
used only for commit/budget accounting, never as a physical residency proxy.
"""
from dataclasses import asdict, dataclass, replace
import math
from pathlib import Path

import psutil

from src.dashboard.system import optional, windows_commit

GB = 1e9
SLACK = 256 * 2**20


class ResourceUnavailable(ValueError):
    pass


@dataclass(frozen=True)
class Memory:
    total: int | None
    available: int | None
    commit_available: int | None
    private: int | None
    working_set: int | None


def measure():
    vm = optional(psutil.virtual_memory)
    commit = optional(windows_commit)
    p = optional(lambda: psutil.Process().memory_info())
    return Memory(vm.total if vm else None, vm.available if vm else None,
                  round(commit['headroom_gb'] * GB) if commit else None,
                  getattr(p, 'private', None), p.rss if p else None)


def reserve(total):
    return max(1_000_000_000, round(total * .12)) + SLACK


def require(m):
    if any(v is None or not math.isfinite(v) or v <= 0
           for v in (m.total, m.available, m.commit_available, m.private, m.working_set)):
        raise ResourceUnavailable('Memory measurements unavailable; no safe allocation can be selected. Retry when Windows measurements are available.')
    if m.available > m.total:
        raise ResourceUnavailable('Inconsistent physical RAM measurements; retry.')


def select(m, *, resident, transient, baseline, record, max_cache, minimum,
           resident_loaded=False, allocated_cache=0, profile='auto', budget_gb=3.5, cache_gb=1.2):
    """Plan only future allocations. Empty bootstrap slots get commit credit only.

    Bootstrap slots have never held weights; do not pretend their committed bytes
    are resident and add them back to physical available RAM.
    """
    require(m)
    future = transient + (0 if resident_loaded else resident)
    keep = reserve(m.total)
    # Existing private commitment is already included in system commit headroom.
    # Count it in the reported process plan, not as physical free-memory credit.
    committed_base = m.private + future - allocated_cache
    physical_room = m.available - keep - future
    commit_room = m.commit_available - keep - future + allocated_cache
    capacity = min(max_cache, physical_room, commit_room)
    if profile != 'auto':
        capacity = min(capacity, round(cache_gb * GB), round(budget_gb * GB) - baseline - resident - transient)
    capacity = int(capacity // record) * record
    if capacity < minimum:
        raise ResourceUnavailable('Insufficient measured physical RAM or Windows commit headroom for the model and minimum cache. Close other workloads or retry; no allocation made.')
    if profile != 'auto' and capacity < round(cache_gb * GB) // record * record:
        raise ResourceUnavailable('Requested cache does not fit measured safe headroom. Choose a smaller manual cache or Automatic.')
    base = baseline + resident + transient
    return {'profile': profile, 'budget_gb': max(base + capacity, committed_base + capacity) / GB if profile == 'auto' else budget_gb,
            'requested_cache_gb': cache_gb if profile != 'auto' else None,
            'cache_bytes': capacity, 'cache_slots': capacity // record,
            'reserve_bytes': keep, 'resident_bytes': resident, 'transient_bytes': transient,
            'baseline_bytes': baseline, 'measurements': asdict(m),
            'measurement_stage': 'resident loaded' if resident_loaded else 'before load'}


def components(store, rt):
    from src.engine.runtime.generate import estimate_ram_bytes, model_config, process_baseline_bytes
    from src.engine.runtime.weights import EMBEDDING
    from src.engine.store.layout import read_resident_index
    import json
    c = model_config(store)
    entries = read_resident_index(Path(store) / 'resident_index.json')
    resident = sum(e.region_size for e in entries if e.name != EMBEDDING)
    baseline = process_baseline_bytes()
    fixed = estimate_ram_bytes(store, c, rt.max_seq, 'cached', 'numpy', 0, 'fused')
    record = json.loads((Path(store) / 'store.json').read_text())['expert_layout']['record_size']
    pinned, free = rt.cache_slots_needed(c)
    return dict(resident=resident, transient=fixed - baseline - resident, baseline=baseline,
                record=record, max_cache=c.num_layers * c.num_experts * record,
                minimum=(pinned + free) * record)


def load(store, mode, rt, profile='auto', budget_gb=3.5, cache_gb=1.2):
    """Fresh model load; caller releases the old model before entering here."""
    from src.engine.runtime.generate import load_model
    from src.engine.manager.backend_selection import BackendChoice
    parts = components(store, rt)
    before = measure()
    # Paging maps the weights normally; reserve only incremental private runtime
    # allocations. Never impose the explicit expert cache on Windows residency.
    if mode == 'mmap':
        require(before)
        if min(before.available, before.commit_available) < reserve(before.total) + parts['transient']:
            raise ResourceUnavailable('Insufficient measured headroom for paging runtime allocations; retry.')
        model, info = load_model(store, mode, rt.max_seq, None, BackendChoice('numpy', None, 'Matched CPU comparison'), rt)
        info['profile'] = {'profile': profile, 'budget_gb': None if profile == 'auto' else budget_gb,
                           'cache_bytes': None, 'reserve_bytes': reserve(before.total),
                           'measurements': asdict(before), 'note': 'OS-managed mappings; no system-wide RAM cap'}
        return model, info
    initial = select(before, **parts, profile=profile, budget_gb=budget_gb, cache_gb=cache_gb)
    boot = replace(rt, expert_cache_gb=parts['minimum'] / GB)
    model = None
    try:
        model, info = load_model(store, mode, rt.max_seq, initial['budget_gb'],
                                 BackendChoice('numpy', None, 'Matched CPU comparison'), boot)
        # Resident weights and current process allocations now appear in actual
        # measurements. Only not-yet-created transient allocations remain future.
        actual = measure()
        final = select(actual, **parts, resident_loaded=True,
                       allocated_cache=model.experts.capacity_bytes, profile=profile,
                       budget_gb=budget_gb, cache_gb=cache_gb)
        model.experts.resize(final['cache_bytes'])
        after = measure()
        require(after)
        if min(after.available, after.commit_available) < reserve(after.total):
            raise ResourceUnavailable('Headroom changed during allocation; model released. Retry with less memory.')
        final['actual_cache_bytes'] = model.experts.capacity_bytes
        final['after_allocation'] = asdict(after)
        info['profile'] = final
        info['cache'] = {**info['cache'], 'bytes': model.experts.capacity_bytes, 'slots': model.experts.num_slots}
        return model, info
    except BaseException:
        if model is not None:
            model.experts.close()
        raise


class SafeBudget:
    """Dashboard-only between-token pressure guard and manual live resizing.

    Existing engine LiveBudget remains unchanged for historical adaptation tools.
    Growth uses measured *additional* room; cache capacity is not added to free
    physical RAM. No prompt, KV or cache is retained across public visitors.
    """
    def __init__(self, model, facts, transient):
        self.model, self.facts, self.transient = model, facts, transient
        self.events = []
        self.pending = None
        self.tokens = 0

    def set(self, value):
        self.pending = value

    def __call__(self):
        self.tokens += 1
        m = measure()
        require(m)
        e = self.model.experts
        keep = reserve(m.total)
        shortage = max(keep - m.available, keep - m.commit_available, 0)
        desired = e.capacity_bytes
        budget, self.pending = self.pending, None
        if budget is not None:
            desired = min(self.facts.num_layers * self.facts.num_experts * self.facts.record_bytes,
                          round(budget * GB) - self.facts.base_ram_bytes)
        if shortage:
            desired = min(desired, e.capacity_bytes - shortage - self.facts.record_bytes)
        elif desired > e.capacity_bytes:
            growth = max(0, min(m.available, m.commit_available) - keep - self.transient)
            desired = min(desired, e.capacity_bytes + growth)
        desired = int(desired // self.facts.record_bytes) * self.facts.record_bytes
        if desired < e.min_slots * self.facts.record_bytes:
            raise ResourceUnavailable('Memory pressure exceeded the safe minimum; run stopped and model released.')
        if desired != e.capacity_bytes:
            old = e.num_slots
            e.resize(desired)
            self.events.append({'token': self.tokens, 'budget_gb': budget,
                                'slots_before': old, 'slots_after': e.num_slots,
                                'reason': 'Measured physical/commit headroom and requested budget'})
        if shortage:
            after = measure()
            require(after)
            if min(after.available, after.commit_available) < reserve(after.total):
                raise ResourceUnavailable('Memory pressure persists after shrinking; run stopped and model released.')
