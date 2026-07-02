#!/usr/bin/env python
"""Tests for pyepo.EPO: ModelSense constants.

Fastest layer: pure enum semantics, no solver. Runs first so a broken import
or constant fails before any expensive solver test.
"""

from pyepo.EPO import MAXIMIZE, MINIMIZE, ModelSense


class TestModelSense:
    """ModelSense values and aliases."""

    def test_integer_values(self):
        # losses rely on these exact signs to flip min/max
        assert MINIMIZE == 1
        assert MAXIMIZE == -1

    def test_members_match_enum(self):
        assert ModelSense.MINIMIZE is MINIMIZE
        assert ModelSense.MAXIMIZE is MAXIMIZE
