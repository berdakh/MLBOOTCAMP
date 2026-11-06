"""Shared fixtures. Tests never touch the network or the learner's artifacts."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session", autouse=True)
def isolated_artifacts(tmp_path_factory):
    """Point churnlab at a throwaway artifact directory for the whole run."""
    path = tmp_path_factory.mktemp("artifacts")
    old = os.environ.get("BOOTCAMP_ARTIFACTS")
    os.environ["BOOTCAMP_ARTIFACTS"] = str(path)
    yield path
    if old is None:
        os.environ.pop("BOOTCAMP_ARTIFACTS", None)
    else:
        os.environ["BOOTCAMP_ARTIFACTS"] = old


@pytest.fixture(scope="session")
def features(isolated_artifacts):
    from churnlab.bootstrap import ensure_features

    return ensure_features(verbose=False)


@pytest.fixture(scope="session")
def split(features):
    from churnlab.splits import time_split

    return time_split(features)


@pytest.fixture(scope="session")
def registry(isolated_artifacts):
    from churnlab.bootstrap import ensure_registry

    return ensure_registry(verbose=False)
