#!/usr/bin/env python
"""Tests for pyepo.EPO: ModelSense constants.

Fastest layer: pure enum semantics, no solver. Runs first so a broken import
or constant fails before any expensive solver test.
"""

from pyepo import EPO
from pyepo.EPO import MAXIMIZE, MINIMIZE, ModelSense


class TestModelSense:

    def test_integer_values(self):
        # losses rely on these exact signs to flip min/max
        assert int(MINIMIZE) == 1
        assert int(MAXIMIZE) == -1

    def test_is_int_enum(self):
        # IntEnum members compare equal to their int value
        assert MINIMIZE == 1
        assert MAXIMIZE == -1

    def test_members_match_enum(self):
        assert ModelSense.MINIMIZE is MINIMIZE
        assert ModelSense.MAXIMIZE is MAXIMIZE

    def test_reexported_on_EPO(self):
        assert EPO.MINIMIZE is MINIMIZE
        assert EPO.MAXIMIZE is MAXIMIZE

    def test_distinct(self):
        assert MINIMIZE != MAXIMIZE
