"""Compatibility shims for old model remote code running on modern transformers.

Qwen-1.0 remote code (modeling_qwen.py) declares ``transformers_stream_generator``
as a hard import, and that package imports beam-constraint classes that
transformers 5.x removed. The streaming-generation path is never used here, so
inert placeholders are enough to let the dynamic-module import check pass.
Importing this module installs the shim.
"""

import transformers
import transformers.generation.utils as _generation_utils

_REMOVED_TOP_LEVEL = (
    "DisjunctiveConstraint",
    "BeamSearchScorer",
    "PhrasalConstraint",
    "ConstrainedBeamSearchScorer",
)

for _name in _REMOVED_TOP_LEVEL:
    if not hasattr(transformers, _name):
        setattr(transformers, _name, type(_name, (), {}))

if not hasattr(_generation_utils, "SampleOutput"):
    _generation_utils.SampleOutput = _generation_utils.GenerateOutput
