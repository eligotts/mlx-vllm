"""
Test script demonstrating LoRA hot-swap MID-GENERATION.

This script:
1. Loads the actual model with LoRA (like a trainer would)
2. Extracts real LoRA weights from the model
3. Starts long-running generation requests on the server
4. WHILE generation is happening, corrupts the LoRA weights
5. Shows how generation degenerates mid-stream

Run with: python tests/test_lora_hotswap.py
"""

import base64
import os
import signal
import subprocess
import sys
import threading
import time

import httpx
import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.tuner.utils import linear_to_lora_layers
from safetensors.numpy import save as save_safetensors


SERVER_URL = "http://localhost:8000"

# Model path (same as server uses)
MODEL_PATH = "/Users/eligottlieb/.lmstudio/models/lmstudio-community/Qwen2.5-7B-Instruct-MLX-4bit"

# LoRA config (must match server's config)
LORA_RANK = 8
LORA_LAYERS = 16
LORA_SCALE = 20.0

# Long prompts that will generate many tokens
LONG_PROMPTS = [
    "Write a detailed essay about the history of artificial intelligence, starting from its origins in the 1950s through modern deep learning. Include key figures, breakthroughs, and future predictions.",
    "Explain the complete process of photosynthesis in plants, including the light-dependent reactions, the Calvin cycle, and how environmental factors affect the process. Be very thorough.",
    "Describe the plot of a fantasy novel about a young wizard discovering their powers. Include character development, world-building, and at least three major plot points.",
]


def wait_for_server(url: str, timeout: float = 60.0) -> bool:
    """Wait for server to be ready."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = httpx.get(f"{url}/health", timeout=2.0)
            if resp.status_code == 200:
                return True
        except httpx.RequestError:
            pass
        time.sleep(1.0)
    return False


def generate_completion_sync(prompt: str, max_tokens: int, label: str) -> str:
    """
    Generate a completion synchronously.
    Returns the full response.
    """
    print(f"\n[{label}] Starting generation...")
    start = time.time()

    resp = httpx.post(
        f"{SERVER_URL}/v1/chat/completions",
        json={
            "model": "test",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        },
        timeout=120.0,
    )
    resp.raise_for_status()

    elapsed = time.time() - start
    content = resp.json()["choices"][0]["message"]["content"]
    print(f"[{label}] Completed in {elapsed:.1f}s, {len(content)} chars")
    return content


def load_model_with_lora() -> dict[str, np.ndarray]:
    """
    Load the actual model with LoRA attached (like a trainer would).
    Returns the LoRA weights as numpy arrays.
    """
    print(f"Loading model from {MODEL_PATH}...")
    model, tokenizer = load(MODEL_PATH)

    print(f"Attaching LoRA (rank={LORA_RANK}, layers={LORA_LAYERS})...")
    lora_config = {
        "rank": LORA_RANK,
        "scale": LORA_SCALE,
        "dropout": 0.0,
    }
    linear_to_lora_layers(model, LORA_LAYERS, lora_config)
    model.eval()

    # Extract LoRA weights (trainable parameters)
    adapter_weights = dict(tree_flatten(model.trainable_parameters()))
    print(f"Extracted {len(adapter_weights)} LoRA parameters")

    # Show some parameter names for debugging
    param_names = list(adapter_weights.keys())
    if param_names:
        print(f"Sample parameter names: {param_names[:4]}")

    # Convert to numpy for serialization
    weights_np = {k: np.array(v) for k, v in adapter_weights.items()}

    # Clean up model to free memory
    del model, tokenizer, adapter_weights
    mx.clear_cache()

    return weights_np


def corrupt_weights(weights: dict[str, np.ndarray], corruption_factor: float = 100.0) -> dict[str, np.ndarray]:
    """Corrupt LoRA weights with large random noise."""
    corrupted = {}
    for key, value in weights.items():
        noise = np.random.randn(*value.shape).astype(value.dtype) * corruption_factor
        corrupted[key] = value + noise
    return corrupted


def zero_weights(weights: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Zero out all LoRA weights (effectively disabling the adapter)."""
    return {k: np.zeros_like(v) for k, v in weights.items()}


def send_lora_weights(weights: dict[str, np.ndarray], version: int) -> dict:
    """Send LoRA weights to the server."""
    # Serialize to safetensors
    weight_bytes = save_safetensors(weights)

    # Base64 encode
    weights_b64 = base64.b64encode(weight_bytes).decode("utf-8")

    # Send HTTP request
    resp = httpx.post(
        f"{SERVER_URL}/adapters/load",
        json={"weights": weights_b64, "version": version},
        timeout=30.0,
    )
    resp.raise_for_status()
    result = resp.json()

    print(f"Loaded adapter v{result['version']} ({len(weights)} tensors, {len(weight_bytes)} bytes)")

    return result


def get_adapter_version() -> int:
    """Get current adapter version from server."""
    resp = httpx.get(f"{SERVER_URL}/adapters/version", timeout=5.0)
    resp.raise_for_status()
    return resp.json()["version"]


def run_mid_flight_test():
    """
    Test that demonstrates LoRA updates happening MID-GENERATION.

    We start multiple long generations, then corrupt the LoRA weights
    while they're still generating. The later parts of the response
    should be affected by the corruption.
    """
    print("=" * 70)
    print("LoRA Hot-Swap MID-FLIGHT Test")
    print("=" * 70)

    # Check server
    print("\n[1] Checking server connection...")
    try:
        resp = httpx.get(f"{SERVER_URL}/health", timeout=5.0)
        if resp.status_code != 200:
            print(f"Server not healthy: {resp.json()}")
            return
        print(f"Server ready: {resp.json()}")
    except httpx.RequestError:
        print(f"Cannot connect to server at {SERVER_URL}")
        print(f"\nStart the server with:")
        print("  MLX_VLLM_LORA_RANK=8 MLX_VLLM_LORA_LAYERS=16 \\ ")
        print("  .venv/bin/python -m uvicorn mlx_vllm.server:app")
        return

    # Load model with LoRA and extract real weights
    print("\n[2] Loading model with LoRA to extract real weights...")
    clean_weights = load_model_with_lora()
    print(f"Got {len(clean_weights)} parameters from model")

    # Initialize server with clean LoRA weights
    print("\n[3] Initializing server with clean LoRA weights...")
    result = send_lora_weights(clean_weights, version=1)
    print(f"Loaded clean adapter v{result['version']}")

    # =========================================================================
    # TEST 1: Baseline - normal generation without mid-flight update
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 1: Baseline generation (no mid-flight update)")
    print("=" * 70)

    baseline_response = generate_completion_sync(
        LONG_PROMPTS[0][:200],  # Shorter prompt for faster test
        max_tokens=256,
        label="Baseline"
    )
    print(f"\nBaseline output preview:\n{baseline_response[:500]}...")

    # =========================================================================
    # TEST 2: Mid-flight corruption
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 2: Mid-flight LoRA corruption")
    print("This will start generation, then CORRUPT weights mid-way")
    print("=" * 70)

    # Reset to clean weights first
    send_lora_weights(clean_weights, version=10)
    print("Reset to clean weights (v10)")

    generation_result = {"response": None, "error": None}

    def background_generation():
        try:
            generation_result["response"] = generate_completion_sync(
                LONG_PROMPTS[0],
                max_tokens=512,  # Long generation
                label="MidFlight"
            )
        except Exception as e:
            generation_result["error"] = str(e)

    # Start generation in background
    print("\nStarting long generation in background...")
    gen_thread = threading.Thread(target=background_generation)
    gen_thread.start()

    # Wait a bit for generation to start, then corrupt
    time.sleep(2.0)  # Let some tokens generate with clean weights

    print("\n>>> CORRUPTING LORA WEIGHTS NOW <<<")
    corrupted = corrupt_weights(clean_weights, corruption_factor=100.0)
    result = send_lora_weights(corrupted, version=11)

    # Wait for generation to complete
    gen_thread.join()

    if generation_result["error"]:
        print(f"Generation error: {generation_result['error']}")
    else:
        mid_flight_response = generation_result["response"]
        print(f"\nMid-flight corrupted output:\n{mid_flight_response}")

    # =========================================================================
    # TEST 3: Multiple concurrent generations with mid-flight update
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 3: Multiple concurrent generations + mid-flight corruption")
    print("=" * 70)

    # Reset to clean
    send_lora_weights(clean_weights, version=20)
    print("Reset to clean weights (v20)")

    results = {}

    def gen_worker(idx: int, prompt: str):
        try:
            results[idx] = generate_completion_sync(
                prompt,
                max_tokens=512,
                label=f"Gen-{idx}"
            )
        except Exception as e:
            results[idx] = f"ERROR: {e}"

    # Start 3 concurrent generations
    print("\nStarting 3 concurrent long generations...")
    threads = []
    for i, prompt in enumerate(LONG_PROMPTS):
        t = threading.Thread(target=gen_worker, args=(i, prompt))
        threads.append(t)
        t.start()
        time.sleep(0.25)  # Stagger starts slightly

    # Wait a bit then corrupt
    time.sleep(1.0)

    print("\n>>> HEAVILY CORRUPTING LORA WEIGHTS <<<")
    heavily_corrupted = corrupt_weights(clean_weights, corruption_factor=500.0)
    result = send_lora_weights(heavily_corrupted, version=21)

    # Wait for all generations
    for t in threads:
        t.join()

    # Print results
    print("\n" + "-" * 70)
    print("RESULTS FROM CONCURRENT GENERATIONS:")
    print("-" * 70)
    for idx in sorted(results.keys()):
        response = results[idx]
        print(f"\n[Gen-{idx}] Output ({len(response)} chars):")
        print(response)

    # =========================================================================
    # TEST 4: Rapid weight updates during generation
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 4: Rapid weight updates during generation")
    print("=" * 70)

    # Reset
    send_lora_weights(clean_weights, version=30)

    rapid_result = {"response": None}

    def rapid_gen():
        rapid_result["response"] = generate_completion_sync(
            "Count from 1 to 100, writing out each number. One, two, three, four...",
            max_tokens=512,
            label="RapidUpdate"
        )

    gen_thread = threading.Thread(target=rapid_gen)
    gen_thread.start()

    # Rapidly alternate between clean and corrupted weights
    for i in range(5):
        time.sleep(1.0)
        if i % 2 == 0:
            corrupted = corrupt_weights(clean_weights, corruption_factor=200.0)
            result = send_lora_weights(corrupted, version=31 + i)
        else:
            result = send_lora_weights(clean_weights, version=31 + i)

    gen_thread.join()

    print(f"\nRapid update output:\n{rapid_result['response']}")

    # Final summary
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
    print(f"Final adapter version: {get_adapter_version()}")
    print("\nIf the mid-flight updates worked, you should see:")
    print("- Early parts of responses are coherent (clean LoRA)")
    print("- Later parts become garbled/repetitive (corrupted LoRA)")
    print("- The transition happens within the SAME response")


def run_with_server():
    """Start server and run test."""
    print("Starting server with LoRA enabled...")

    env = os.environ.copy()
    env["MLX_VLLM_LORA_RANK"] = str(LORA_RANK)
    env["MLX_VLLM_LORA_LAYERS"] = str(LORA_LAYERS)
    env["PYTHONPATH"] = "src"

    server_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "mlx_vllm.server:app",
         "--host", "0.0.0.0", "--port", "8000"],
        env=env,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )

    try:
        print("Waiting for server to start...")
        if not wait_for_server(SERVER_URL, timeout=180.0):
            print("Server failed to start within timeout")
            return

        print("Server is ready!\n")
        run_mid_flight_test()

    finally:
        print("\nShutting down server...")
        server_proc.send_signal(signal.SIGTERM)
        try:
            server_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server_proc.kill()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test LoRA hot-swap mid-flight")
    parser.add_argument(
        "--start-server",
        action="store_true",
        help="Start the server automatically",
    )
    args = parser.parse_args()

    if args.start_server:
        run_with_server()
    else:
        run_mid_flight_test()
