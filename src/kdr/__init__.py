"""kdr — Knowledge Distillation Recovery.

Unified BF16 KD + DA-QAD trainer. Mode flag drives whether `mtq.quantize` is
installed before the FKLD loop. Output is HF compressed-tensors safetensors.

See `requirements/` for the HLR/LLR set.
"""

__version__ = "0.1.0"
