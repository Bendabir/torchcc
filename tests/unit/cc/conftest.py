from __future__ import annotations

import pytest
import torch


@pytest.fixture(scope="session")
def generator(device: torch.device) -> torch.Generator:
    g = torch.Generator(device)

    g.manual_seed(123456789)

    return g
