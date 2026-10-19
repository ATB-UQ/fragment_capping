from importlib import reload

import fragment_capping.config as config


def test_failed_ilp_debug_path_is_off_by_default(monkeypatch):
    monkeypatch.delenv('ATB_FRAGMENT_CAPPING_DEBUG', raising=False)
    reload(config)
    assert config.failed_ilp_debug_path('x_debug.lp') is None


def test_failed_ilp_debug_path_never_uses_working_directory(monkeypatch, tmp_path):
    monkeypatch.setenv('ATB_FRAGMENT_CAPPING_DEBUG', 'true')
    monkeypatch.setenv('ATB_FRAGMENT_CAPPING_DEBUG_DIR', str(tmp_path / 'dbg'))
    reload(config)
    path = config.failed_ilp_debug_path('x_debug.lp')
    assert path == str(tmp_path / 'dbg' / 'x_debug.lp')
    assert (tmp_path / 'dbg').is_dir()
    # force=True (caller-supplied debug stream) still redirects out of the cwd
    monkeypatch.delenv('ATB_FRAGMENT_CAPPING_DEBUG')
    reload(config)
    assert config.failed_ilp_debug_path('y.lp') is None
    assert config.failed_ilp_debug_path('y.lp', force=True).endswith('y.lp')


def test_sequential_solve_within_raises_when_budget_exhausted():
    from pulp import LpProblem, LpMinimize, LpVariable
    import pytest
    reload(config)
    problem = LpProblem('t', LpMinimize)
    x = LpVariable('x', 0, 1, cat='Integer')
    problem += x
    with pytest.raises(config.ILP_Timeout):
        config.sequential_solve_within(problem, [x], timeout=0.5)
