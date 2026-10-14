import pytest

from dorsal.analytics.financials import analyze
from dorsal.analytics.forecast import Forecaster
from dorsal.runtime import DorsalRuntime
from dorsal.sandbox.generator import harbor_lane


@pytest.fixture
def hero():
    biz, truth = harbor_lane()
    return biz, truth


@pytest.fixture
def fc(hero):
    return Forecaster(hero[0], n_paths=400, buffer=12000)


@pytest.fixture
def fin(hero):
    return analyze(hero[0])


@pytest.fixture
def rt():
    biz, _ = harbor_lane()
    return DorsalRuntime(biz, n_paths=400)
