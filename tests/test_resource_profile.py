from dataclasses import replace
from types import SimpleNamespace
import pytest

from src.dashboard.resources import Memory, ResourceUnavailable, SafeBudget, reserve, select

PARTS = dict(resident=1_600_000_000, transient=400_000_000, baseline=300_000_000,
             record=8_671_232, max_cache=12_486_574_080, minimum=13 * 8_671_232)


def machine(total=16_000_000_000, available=14_000_000_000, commit=18_000_000_000):
    return Memory(total, available, commit, 250_000_000, 200_000_000)


def test_8gb_and_16gb_choose_machine_sized_cache():
    small = select(machine(8_000_000_000, 5_000_000_000, 8_000_000_000), **PARTS)
    large = select(machine(), **PARTS)
    assert 1e9 < small['cache_bytes'] < 2e9
    assert 9e9 < large['cache_bytes'] < 11e9
    assert large['cache_bytes'] > small['cache_bytes'] * 4
    assert large['cache_bytes'] % PARTS['record'] == 0
    assert large['cache_bytes'] + PARTS['resident'] + PARTS['transient'] + reserve(16_000_000_000) <= 14_000_000_000


@pytest.mark.parametrize('measurement', [machine(available=2_000_000_000), machine(commit=2_000_000_000),
                                         replace(machine(), available=None), replace(machine(), commit_available=None),
                                         replace(machine(), working_set=None)])
def test_low_or_missing_measurements_refuse(measurement):
    with pytest.raises(ResourceUnavailable):
        select(measurement, **PARTS)


def test_resident_load_not_subtracted_twice_and_private_is_not_physical():
    m = machine()
    a = select(m, **PARTS)
    # Resident pages/commit now consumed: remove them from the future allocation.
    b = select(replace(m, available=m.available-PARTS['resident'],
                       commit_available=m.commit_available-PARTS['resident']-PARTS['minimum']),
               **PARTS, resident_loaded=True, allocated_cache=PARTS['minimum'])
    assert b['cache_bytes'] == a['cache_bytes']
    c = select(replace(m, private=8_000_000_000, working_set=1_000_000_000), **PARTS)
    assert c['cache_bytes'] == a['cache_bytes']  # no private-byte credit to RAM
    assert c['budget_gb'] > a['budget_gb']  # existing commitment is reported
    assert select(m, **PARTS, budget_gb=.5, cache_gb=.05)['cache_bytes'] == a['cache_bytes']


def test_manual_constrained_settings_preserved_and_unsafe_request_refused():
    for name in ('manual', 'constrained'):
        p = select(machine(), **PARTS, profile=name, budget_gb=3.5, cache_gb=1.2)
        assert p['cache_bytes'] == 138 * PARTS['record'] and p['budget_gb'] == 3.5
    with pytest.raises(ResourceUnavailable):
        select(machine(available=3_000_000_000), **PARTS, profile='manual', cache_gb=1.2)


def test_live_growth_does_not_add_existing_cache_to_free_physical_ram(monkeypatch):
    import src.dashboard.resources as resources
    record = PARTS['record']
    class Experts:
        capacity_bytes = 138 * record
        num_slots = 138
        min_slots = 13
        def resize(self, n):
            self.capacity_bytes = n
            self.num_slots = n // record
            return self.num_slots
    experts = Experts()
    model = SimpleNamespace(experts=experts)
    facts = SimpleNamespace(num_layers=24, num_experts=60, record_bytes=record, base_ram_bytes=2_300_000_000)
    m = machine(available=reserve(16_000_000_000)+PARTS['transient']+100_000_000)
    monkeypatch.setattr(resources, 'measure', lambda: m)
    live = SafeBudget(model, facts, PARTS['transient'])
    old = experts.capacity_bytes
    live.set(10)
    live()
    assert experts.capacity_bytes <= old+100_000_000
    live.set(2.8)
    live()
    assert experts.capacity_bytes < old and len(live.events) == 2


def test_two_stage_loader_rechecks_and_cleans_up_on_headroom_loss(monkeypatch):
    import src.dashboard.resources as resources
    import src.engine.runtime.generate as generate
    class Experts:
        capacity_bytes = PARTS['minimum']
        num_slots = 13
        closed = False
        def resize(self, n):
            self.capacity_bytes, self.num_slots = n, n//PARTS['record']
        def close(self):
            self.closed = True
    experts = Experts()
    monkeypatch.setattr(resources, 'components', lambda *a: PARTS)
    monkeypatch.setattr(generate, 'load_model', lambda *a: (SimpleNamespace(experts=experts), {'cache': {}}))
    measurements = iter([machine(), machine(available=1_000_000_000)])
    monkeypatch.setattr(resources, 'measure', lambda: next(measurements))
    rt = generate.RuntimeConfig.load(__import__('pathlib').Path('configs/runtime_cache_chat.json'))
    with pytest.raises(ResourceUnavailable):
        resources.load(None, 'cached', rt)
    assert experts.closed
