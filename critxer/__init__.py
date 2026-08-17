"""critxer --- does an auditor's future repair role change its present diagnosis?

A responsibility ladder over false-alarm rates on ProcessBench: from a sentence stating a repair
will follow, through an obligation discharged in the same conversation, to a completed audit ->
repair episode already in the model's context.

* :mod:`critxer.core` -- the library: prompt assembly, metrics, resampling, multiplicity, HTTP
  client. No absolute paths, no argument parsing, no import-time side effects, all unit-tested.
* :mod:`critxer.cli` -- one module per command, grouped by pipeline stage.

``python -m critxer`` lists the commands.
"""

__version__ = "0.2.0"
