"""Tests for the ``--layers`` and ``--max-layers`` selectors."""

from __future__ import annotations

import pytest

from flatrun.cli import _parse_layer_spec, _select_layers
from flatrun.utils.errors import ConfigurationError
from flatrun.utils.types import LayerDescriptor


@pytest.fixture()
def manifest_40() -> tuple:
    """A 40-layer synthetic manifest."""
    return tuple(
        LayerDescriptor(index=i, tensor_names=(f"L{i}.w",))
        for i in range(40)
    )


# Pure list
def test_parse_layer_spec_pure_list() -> None:
    assert _parse_layer_spec("1,3,4,6,7,8") == [1, 3, 4, 6, 7, 8]


# Pure range
def test_parse_layer_spec_pure_range() -> None:
    assert _parse_layer_spec("0-6") == [0, 1, 2, 3, 4, 5, 6]


# Mixed list + range
def test_parse_layer_spec_mixed() -> None:
    assert _parse_layer_spec("1,3-4,6,7-8") == [1, 3, 4, 6, 7, 8]


# Multiple ranges
def test_parse_layer_spec_multiple_ranges() -> None:
    assert _parse_layer_spec("0-6,19-24,34-39") == (
        list(range(7)) + list(range(19, 25)) + list(range(34, 40))
    )


# Whitespace
def test_parse_layer_spec_whitespace() -> None:
    assert _parse_layer_spec(" 1 , 3 - 4 , 6 ") == [1, 3, 4, 6]


# Single element
def test_parse_layer_spec_single() -> None:
    assert _parse_layer_spec("5") == [5]


# Degenerate single-element range
def test_parse_layer_spec_single_as_range() -> None:
    assert _parse_layer_spec("5-5") == [5]


# Dedup across list and range
def test_parse_layer_spec_dedup() -> None:
    assert _parse_layer_spec("1,2,1,3,2") == [1, 2, 3]


# Validation errors
def test_parse_layer_spec_reversed_range() -> None:
    with pytest.raises(ConfigurationError, match="reversed"):
        _parse_layer_spec("5-3")


def test_parse_layer_spec_malformed_range() -> None:
    with pytest.raises(ConfigurationError, match="malformed"):
        _parse_layer_spec("1-")


def test_parse_layer_spec_empty() -> None:
    with pytest.raises(ConfigurationError, match="non-empty"):
        _parse_layer_spec("")


def test_parse_layer_spec_non_int() -> None:
    with pytest.raises(ConfigurationError, match="not an integer"):
        _parse_layer_spec("abc")


def test_parse_layer_spec_empty_entry() -> None:
    with pytest.raises(ConfigurationError, match="empty entry"):
        _parse_layer_spec("1,,2")


# End-to-end with manifest
def test_select_layers_range_expansion(manifest_40) -> None:
    sel = _select_layers(manifest_40, max_layers=None, layers_spec="0-6,19-24,34-39")
    assert [l.index for l in sel] == (
        list(range(7)) + list(range(19, 25)) + list(range(34, 40))
    )


def test_select_layers_range_out_of_bounds(manifest_40) -> None:
    with pytest.raises(ConfigurationError, match="not in the manifest"):
        _select_layers(manifest_40, max_layers=None, layers_spec="0-100")


def test_select_layers_single_int_single_range(manifest_40) -> None:
    assert [l.index for l in _select_layers(manifest_40, max_layers=None, layers_spec="5-5")] == [5]
    assert [l.index for l in _select_layers(manifest_40, max_layers=None, layers_spec="5")] == [5]


def test_select_layers_mixed_list_and_range_preserves_order(manifest_40) -> None:
    sel = _select_layers(manifest_40, max_layers=None, layers_spec="3-4,1,7-8")
    # The order in the spec is what matters: 3,4,1,7,8 - but 1
    # comes after 3,4 and the dedup keeps first-seen order.
    assert [l.index for l in sel] == [3, 4, 1, 7, 8]


def test_select_layers_mutually_exclusive(manifest_40) -> None:
    with pytest.raises(ConfigurationError, match="mutually exclusive"):
        _select_layers(manifest_40, max_layers=5, layers_spec="1,2")


def test_select_layers_max_layers_basic(manifest_40) -> None:
    sel = _select_layers(manifest_40, max_layers=3, layers_spec=None)
    assert [l.index for l in sel] == [0, 1, 2]


def test_select_layers_no_args_returns_all(manifest_40) -> None:
    sel = _select_layers(manifest_40, max_layers=None, layers_spec=None)
    assert len(sel) == 40
