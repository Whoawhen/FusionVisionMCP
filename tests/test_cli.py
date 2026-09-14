#  test_cli.py
#
#  Copyright (c) 2025-2026 Junpei Kawamoto
#
#  This software is released under the MIT License.
#
#  http://opensource.org/licenses/mit-license.php
import pytest
import rich_click as click

from fusion_vision_mcp.cli import DEFAULT_MEMORY_MODE, resolve_memory_mode


def test_persistent_never_releases() -> None:
    """A timeout of 0 is what `IdleReleased` reads as "never schedule a release"."""
    assert resolve_memory_mode("persistent") == 0.0


@pytest.mark.parametrize(("mode", "expected"), [("aggressive", 5.0), ("standard", 10.0)])
def test_named_presets_map_to_their_documented_durations(mode: str, expected: float) -> None:
    assert resolve_memory_mode(mode) == expected


def test_the_default_mode_is_a_finite_timeout() -> None:
    """The out-of-the-box setting has to actually give memory back, not hold it forever."""
    assert resolve_memory_mode(DEFAULT_MEMORY_MODE) > 0


@pytest.mark.parametrize("value", ["30", "2.5", "0"])
def test_a_bare_number_is_taken_as_user_defined_minutes(value: str) -> None:
    assert resolve_memory_mode(value) == float(value)


@pytest.mark.parametrize("value", ["  Standard  ", "PERSISTENT"])
def test_modes_are_case_and_whitespace_insensitive(value: str) -> None:
    assert resolve_memory_mode(value) == resolve_memory_mode(value.strip().lower())


@pytest.mark.parametrize("value", ["turbo", "", "-5"])
def test_unusable_values_are_rejected_with_a_message_naming_the_option(value: str) -> None:
    with pytest.raises(click.BadParameter) as excinfo:
        resolve_memory_mode(value)

    # `format_message` is what click prints to the user; `str()` drops the param hint.
    assert "--memory-mode" in excinfo.value.format_message()
