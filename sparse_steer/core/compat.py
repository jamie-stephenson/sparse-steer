"""Compatibility shims for old model remote code running on modern transformers.

Qwen-1.0 remote code (modeling_qwen.py) declares ``transformers_stream_generator``
as a hard import, and that package imports beam-constraint classes that
transformers 5.x removed. The streaming-generation path is never used here, so
inert placeholders are enough to let the dynamic-module import check pass.
Importing this module installs the shim.
"""

# transformer_lens replaces sys.modules["transformers"] on import, discarding any
# attributes patched onto the old module object — so it must be imported first.
import transformer_lens  # noqa: F401

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

# transformers 5.x removed the head-pruning API; modeling_qwen.py calls
# get_head_mask unconditionally in forward. Only the head_mask=None path is used.
if not hasattr(transformers.PreTrainedModel, "get_head_mask"):

    def _get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
        if head_mask is not None:
            raise NotImplementedError("get_head_mask shim only supports head_mask=None")
        return [None] * num_hidden_layers

    transformers.PreTrainedModel.get_head_mask = _get_head_mask
