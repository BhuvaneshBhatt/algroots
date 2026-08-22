"""Shared pytest configuration for algroots."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _disable_unrelated_auto_recognition(request, monkeypatch):
    """Keep non-recognition tests focused and fast.

    Automatic recognition is a public default, but invoking FLINT/LLL after every
    numerical solve makes unrelated solver, property, and documentation tests both
    slow and coupled to the recognizer. Tests marked ``recognition`` exercise the
    real/default recognition path; all others replace only the automatic
    post-processing hook. Explicit ``recognize_system_roots`` calls are unaffected.
    """
    if request.node.get_closest_marker("recognition") is not None:
        return

    from algroots import recognition as recognition_module

    monkeypatch.setattr(
        recognition_module,
        "_auto_recognize",
        lambda result, *, recognize, max_degree: result,
    )
