from __future__ import annotations

import pytest
import torch


@pytest.fixture(scope="session")
def generator(device: torch.device, seed: int) -> torch.Generator:
    g = torch.Generator(device)

    g.manual_seed(seed)

    return g
