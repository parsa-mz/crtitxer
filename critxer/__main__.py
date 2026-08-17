"""Command dispatch: ``critxer <command> [args...]``.

Thin by design. Each command owns its own ``argparse`` parser, so this resolves a name to a module,
hands the remaining argv over untouched, and does nothing else -- which keeps a command's ``--help``
identical whether it is reached through here or by importing the module directly.

Commands are grouped in the order the pipeline runs, because that order is not obvious from the
names: allocation precedes generation, generation precedes analysis, and the screen precedes all.
"""

from __future__ import annotations

import importlib
import sys

# name -> (module under critxer.cli, summary). Blank-line groups are reproduced in --help.
COMMANDS: dict[str, tuple[str, str]] = {
    # Setup
    "allocate": ("make.allocation", "draw the disjoint source-stratified arms"),
    "power-sweep": ("run.power_sweep", "sample-size and null-calibration simulation"),
    "kill-tests": ("run.kill_tests", "pre-specified falsification checks"),
    # Screening: a flat ladder on an insensitive instrument is not a result
    "bakeoff": ("run.bakeoff", "candidate-model shortlist on a small subset"),
    "screen": ("run.pc_diagnostic", "gate 0 instrument-sensitivity screen (PC/PCL/PCH)"),
    "analyse-screen": ("analyse.gate0", "score the screen and report the framing range"),
    "calibrate-gate2": ("run.gate2_calibration", "parse-rate and budget calibration"),
    # Generation
    "run-ladder": ("run.ladder", "the R0-R3u rungs on the clean arm"),
    "run-r4": ("run.r4", "the prior-context 2x2 and its five controls"),
    "run-episodes": ("run.warmup_episodes", "generate and freeze episodes, fillers, continuations"),
    "run-detection": ("run.detection", "the labelled-incorrect arm"),
    # Fault injection, which supplies the incorrect-verdict episodes
    "inject-pilot": ("run.injection_pilot", "try the injector on a handful of traces"),
    "inject": ("run.injection_set", "build the validated early/late fault set"),
    "inject-calibrate": ("run.position_calibration", "calibrate fault position"),
    "inject-detectability": ("run.position_detectability", "early-vs-late detectability gap"),
    # Analysis
    "analyse-ladder": ("analyse.ladder", "rung-vs-R0 contrasts, N2 band, Holm"),
    "analyse-r4": ("analyse.r4", "the factorial, its controls, and the repair decomposition"),
    "analyse-templates": ("analyse.templates", "one contrast across all five wordings"),
    "analyse-detection": ("analyse.detection", "d', criterion, localisation, ECE, Holm"),
    "figures": ("make.figures", "the paper's two figures from persisted JSON"),
    "tables": ("make.tables", "the paper's artefact-driven LaTeX tables and the prose counts"),
    # Maintenance
    "backfill-episode-ids": ("tools.backfill_episode_ids",
                             "recover pairings for pre-fingerprint cells"),
    "dump-audit-sample": ("tools.dump_manual_audit_sample",
                          "stratified sample for hand inspection"),
}


def usage() -> str:
    width = max(len(c) for c in COMMANDS)
    lines = ["usage: critxer <command> [args...]", "", "commands:"]
    for name, (_module, summary) in COMMANDS.items():
        lines.append(f"  {name:<{width}}  {summary}")
    lines += ["", "Run `critxer <command> --help` for a command's own options.",
              "Artefacts are read and written under $CRITXER_DATA (see critxer.core.paths)."]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(usage())
        return 0
    name, rest = argv[0], argv[1:]
    if name not in COMMANDS:
        print(f"critxer: unknown command {name!r}\n\n{usage()}", file=sys.stderr)
        return 2

    module = importlib.import_module(f"critxer.cli.{COMMANDS[name][0]}")
    # The command parses argv itself, so hand it over as if it had been invoked directly. Rewriting
    # sys.argv[0] to the dispatched name keeps its --help output honest.
    sys.argv = [f"critxer {name}", *rest]
    entry = module.main
    result = entry()
    # Async commands (anything that talks to a served model) return a coroutine from main().
    if hasattr(result, "__await__"):
        import asyncio

        asyncio.run(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
