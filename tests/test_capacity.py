from api import capacity
from config import settings


def test_generation_capacity_rejects_excess_and_recovers(monkeypatch):
    monkeypatch.setattr(settings, "MAX_CONCURRENT_GENERATIONS", 1)
    capacity.reset_generation_capacity()

    assert capacity.try_acquire_generation_slot() is True
    assert capacity.try_acquire_generation_slot() is False

    capacity.release_generation_slot()
    assert capacity.try_acquire_generation_slot() is True
    capacity.release_generation_slot()
