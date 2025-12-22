"""CLI entrypoint for MLX-vLLM server."""

import argparse

import uvicorn

from mlx_vllm.config import ServerConfig
from mlx_vllm.server import set_config


def main() -> None:
    """Run the MLX-vLLM server."""
    parser = argparse.ArgumentParser(description="MLX-vLLM OpenAI-compatible server")
    parser.add_argument("--model", type=str, help="Model path or HuggingFace repo ID")
    parser.add_argument("--host", type=str, help="Host to bind to")
    parser.add_argument("--port", type=int, help="Port to bind to")
    parser.add_argument("--max-batch-size", type=int, help="Maximum batch size")
    parser.add_argument("--max-tokens", type=int, help="Default max tokens per request")
    args = parser.parse_args()

    # Build config, overriding with CLI args if provided
    config_overrides = {}
    if args.model:
        config_overrides["model_path"] = args.model
    if args.host:
        config_overrides["host"] = args.host
    if args.port:
        config_overrides["port"] = args.port
    if args.max_batch_size:
        config_overrides["max_batch_size"] = args.max_batch_size
    if args.max_tokens:
        config_overrides["max_tokens"] = args.max_tokens

    config = ServerConfig(**config_overrides)
    set_config(config)

    print("Starting MLX-vLLM server")
    print(f"  Model: {config.model_path}")
    print(f"  Host: {config.host}:{config.port}")
    print(f"  Max batch size: {config.max_batch_size}")
    print(f"  Default max tokens: {config.max_tokens}")

    uvicorn.run(
        "mlx_vllm.server:app",
        host=config.host,
        port=config.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
