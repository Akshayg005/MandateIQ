"""src/llm/tools.py -- tool definitions for forced function calling."""
from __future__ import annotations

import re

import pytest

from src.core.types import DeclineClass
from src.llm.tools import INTENT_TOOL, NORMALIZE_TOOL


def test_normalize_tool_has_required_structure():
    """NORMALIZE_TOOL is a valid function-call declaration."""
    assert NORMALIZE_TOOL["name"]
    assert NORMALIZE_TOOL["description"]
    assert NORMALIZE_TOOL["parameters"]["type"] == "object"
    assert "properties" in NORMALIZE_TOOL["parameters"]
    assert "required" in NORMALIZE_TOOL["parameters"]


def test_normalize_tool_decline_class_enum_lives():
    """NORMALIZE_TOOL["parameters"]["properties"]["decline_class"]["enum"]
    exists and is non-empty.
    """
    assert "decline_class" in NORMALIZE_TOOL["parameters"]["properties"]
    props = NORMALIZE_TOOL["parameters"]["properties"]["decline_class"]
    assert "enum" in props
    assert len(props["enum"]) > 0


def test_normalize_tool_decline_class_enum_matches_decline_class():
    """The enum is EXACTLY {c.value for c in DeclineClass}, live comparison.
    This is a source-of-truth test: if someone adds an 8th DeclineClass and
    forgets to touch tools.py, the enum silently goes stale and the model
    can never emit the new value.
    """
    expected_enum = {c.value for c in DeclineClass}
    actual_enum = set(NORMALIZE_TOOL["parameters"]["properties"]["decline_class"]["enum"])
    assert actual_enum == expected_enum


def test_normalize_tool_both_fields_required():
    """Both decline_class and confidence are in required."""
    required = NORMALIZE_TOOL["parameters"]["required"]
    assert "decline_class" in required
    assert "confidence" in required
    assert len(required) == 2


def test_intent_tool_has_required_structure():
    """INTENT_TOOL is a valid function-call declaration."""
    assert INTENT_TOOL["name"]
    assert INTENT_TOOL["description"]
    assert INTENT_TOOL["parameters"]["type"] == "object"
    assert "properties" in INTENT_TOOL["parameters"]
    assert "required" in INTENT_TOOL["parameters"]


def test_intent_tool_both_fields_required():
    """Both intent_score and rationale are in required."""
    required = INTENT_TOOL["parameters"]["required"]
    assert "intent_score" in required
    assert "rationale" in required
    assert len(required) == 2


def test_normalize_and_intent_tool_names_are_different():
    """Tool names must not collide (a forced-call config selects by name)."""
    assert NORMALIZE_TOOL["name"] != INTENT_TOOL["name"]


def test_normalize_tool_name_is_string():
    """NORMALIZE_TOOL["name"] is a non-empty string."""
    assert isinstance(NORMALIZE_TOOL["name"], str)
    assert len(NORMALIZE_TOOL["name"]) > 0


def test_intent_tool_name_is_string():
    """INTENT_TOOL["name"] is a non-empty string."""
    assert isinstance(INTENT_TOOL["name"], str)
    assert len(INTENT_TOOL["name"]) > 0


def test_normalize_tool_properties_schema():
    """decline_class property is a valid string enum, confidence is a number."""
    props = NORMALIZE_TOOL["parameters"]["properties"]

    assert props["decline_class"]["type"] == "string"
    assert "enum" in props["decline_class"]

    assert props["confidence"]["type"] == "number"


def test_intent_tool_properties_schema():
    """intent_score property is a number, rationale is a string."""
    props = INTENT_TOOL["parameters"]["properties"]

    assert props["intent_score"]["type"] == "number"
    assert props["rationale"]["type"] == "string"
