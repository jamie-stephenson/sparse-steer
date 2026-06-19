"""Steering subsystem: load a SteeringModel and decide its per-site (direction, strength) config.

- ``experiment.py`` — the SteeringExperiment engine (dispatches the strength solver).
- ``solvers.py``    — field solvers: direction (self/pin = declare) + strength (none/gate_training).
- ``search.py``     — the score-search direction solver (Arditi grid select).
- ``extract.py``    — cache-aware steering-vector extraction (was experiment/_common.py).

``core/steering.py`` is the orthogonal *apply* mechanism; this subsystem only decides the config.
"""

from .experiment import STRENGTH_SOLVERS, SteeringExperiment
from .extract import run_extraction
from .search import candidate_grid, filter_and_pick, grid_select_source
from .solvers import (
    _parse_source,
    broadcast,
    extraction_targets,
    resolve_direction_source,
    source_vectors,
)

__all__ = [
    "STRENGTH_SOLVERS",
    "SteeringExperiment",
    "_parse_source",
    "broadcast",
    "candidate_grid",
    "extraction_targets",
    "filter_and_pick",
    "grid_select_source",
    "resolve_direction_source",
    "run_extraction",
    "source_vectors",
]
