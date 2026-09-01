"""
Shared pytest configuration and fixtures.
"""

import numpy as np
import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks test as slow (skip with -m 'not slow')")
    config.addinivalue_line("markers", "hardware: marks test requiring physical ESP32-S3")
    config.addinivalue_line("markers", "model: marks test requiring TFLite model artifact")


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    """Shared deterministic random number generator."""
    return np.random.default_rng(42)


@pytest.fixture
def unit_embedding(rng: np.random.Generator) -> np.ndarray:
    """A random unit-norm 64-dim embedding."""
    v = rng.standard_normal(64).astype(np.float32)
    return v / np.linalg.norm(v)
