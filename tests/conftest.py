import pytest

from dorsal.analytics.forecast import Forecaster
from dorsal.sandbox.generator import harbor_lane


@pytest.fixture
def hero():
    biz, truth = harbor_lane()
    return biz, truth


@pytest.fixture
def fc(hero):
    return Forecaster(hero[0], n_paths=400, buffer=12000)
