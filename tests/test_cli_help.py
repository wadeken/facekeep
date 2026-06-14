"""The CLI ``--help`` text carries the product positioning: aggressive mode is
the headline, faithful mode is the honest default.

Pure copy regression — these touch no output pixels and assert no behavior, only
that the positioning survives in the help a user actually reads. Assertions are
deliberately token-level (not exact marketing phrasing) so wording can be tuned
without churning the tests, while still failing if a mode or the on-ramp drops
out of the help entirely.
"""

from click.testing import CliRunner

from facekeep.cli import cli


def test_group_help_positions_both_modes():
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0, result.output
    out = result.output.lower()
    assert "aggressive" in out          # the headline mode is named
    assert "faithful" in out            # the default mode is named
    assert "default" in out             # faithful is identified as the default
    assert "preset" in out              # the one-word on-ramp is surfaced


def test_compress_help_describes_both_modes():
    result = CliRunner().invoke(cli, ["compress", "--help"])
    assert result.exit_code == 0, result.output
    out = result.output.lower()
    assert "aggressive" in out
    assert "faithful" in out
    assert ".fkeep" in out              # aggressive's output is explained


def test_mode_default_is_still_faithful():
    """Positioning is copy-only: the bare-`compress` default must stay faithful
    (a backup tool shouldn't silently hand back a reconstructed background)."""
    result = CliRunner().invoke(cli, ["compress", "--help"])
    assert result.exit_code == 0, result.output
    assert "faithful (default)" in result.output.lower()
