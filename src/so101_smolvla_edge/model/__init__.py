"""SmolVLA inference: bundle export/loading, the static-shape model and the CUDA-graph engine."""

from .._cuda_libs import preload as _preload

_preload()  # before any submodule imports torch
