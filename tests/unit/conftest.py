from __future__ import annotations

import pytest
import torch


@pytest.fixture(scope="session")
def device() -> torch.device:
    return torch.device("cuda:0")


@pytest.fixture(scope="session")
def seed() -> int:
    return 123456789
