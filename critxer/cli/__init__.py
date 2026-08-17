"""One module per command, grouped by pipeline stage.

* :mod:`critxer.cli.run` -- talks to a served model and writes generations.
* :mod:`critxer.cli.analyse` -- reads those generations and writes contrasts. No inference.
* :mod:`critxer.cli.make` -- allocation, figures, tables.
* :mod:`critxer.cli.tools` -- maintenance on existing artefacts.

Each command keeps its own argument parsing and its own ``main()``. Commands a later one supersedes
are kept: they produced numbers in the paper, and a reader tracing a figure back to its artefact
needs the command that actually wrote it.

:mod:`critxer.__main__` maps command names to these modules.
"""
