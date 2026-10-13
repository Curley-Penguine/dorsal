import pytest

from dorsal.sandbox.generator import harbor_lane


@pytest.fixture
def hero():
    biz, truth = harbor_lane()
    return biz, truth
