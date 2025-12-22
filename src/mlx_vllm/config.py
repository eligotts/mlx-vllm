"""Server configuration."""

from pydantic_settings import BaseSettings


class ServerConfig(BaseSettings):
    """Configuration for MLX-vLLM server."""

    model_path: str = "/Users/eligottlieb/.lmstudio/models/lmstudio-community/Qwen2.5-7B-Instruct-MLX-4bit"
    host: str = "0.0.0.0"
    port: int = 8000
    max_batch_size: int = 32
    max_tokens: int = 4096

    model_config = {"env_prefix": "MLX_VLLM_"}
