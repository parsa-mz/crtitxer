"""Tests for the command dispatcher.

``COMMANDS`` maps a name a human types to a dotted module path, and nothing else checks that the
mapping still points at anything. A stale entry fails only when someone runs that one command, which
for a superseded command can be months later or never, so this walks every entry.
"""

from __future__ import annotations

import importlib

import pytest

from critxer.__main__ import COMMANDS, main, usage


@pytest.mark.parametrize("name", sorted(COMMANDS))
def test_command_resolves_to_a_module_with_a_main(name):
    module_path, summary = COMMANDS[name]
    module = importlib.import_module(f"critxer.cli.{module_path}")
    assert callable(module.main), f"{module_path}.main is not callable"
    assert summary, f"{name} has no summary, so it is invisible in --help"


def test_every_command_lives_in_a_known_group():
    # The grouping is the point of the subpackages: a command that talks to a model belongs under
    # run/, one that only reads persisted JSON under analyse/. A module left at the cli/ top level
    # would still import and would quietly lose that distinction.
    groups = {"run", "analyse", "make", "tools"}
    for name, (module_path, _) in COMMANDS.items():
        group = module_path.split(".")[0]
        assert group in groups, f"{name} -> {module_path} is outside {sorted(groups)}"


def test_usage_lists_every_command():
    text = usage()
    for name in COMMANDS:
        assert name in text, f"{name} is missing from --help"


def test_unknown_command_exits_nonzero_without_raising(capsys):
    assert main(["no-such-command"]) == 2
    assert "unknown command" in capsys.readouterr().err


@pytest.mark.parametrize("flag", ["-h", "--help", "help"])
def test_help_is_zero_exit(flag, capsys):
    assert main([flag]) == 0
    assert "usage: critxer" in capsys.readouterr().out


def test_no_argv_prints_usage(capsys):
    assert main([]) == 0
    assert "commands:" in capsys.readouterr().out
