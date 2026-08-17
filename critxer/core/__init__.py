"""Library code: pure functions, unit-tested, no I/O beyond the HTTP backend.

The split from :mod:`critxer.cli` is about where correctness is enforced. Anything a wrong answer
could hide in -- prompt assembly, the 0-based/1-based step conversion, the null band, the
two-stage bootstrap, Holm step-down -- lives here and has tests. The CLI modules orchestrate:
they read files, call these, and write JSON.
"""
