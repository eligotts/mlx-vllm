"""MLX-vLLM: OpenAI-compatible inference server using MLX."""

from mlx_vllm.engine import ContinuousBatchingEngine, GenerationOutput

__version__ = "0.1.0"
__all__ = ["ContinuousBatchingEngine", "GenerationOutput"]
