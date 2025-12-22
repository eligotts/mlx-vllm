"""Test script for the OpenAI-compatible API.

Run the server first:
    uv run mlx-vllm --model /path/to/your/model

Then run this test:
    uv run python tests/test_api.py
"""

import httpx

BASE_URL = "http://localhost:8000"


def test_health():
    """Test the health endpoint."""
    print("Testing /health...")
    response = httpx.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    print(f"  Health: {data}")
    print("PASSED: test_health\n")


def test_models():
    """Test the /v1/models endpoint."""
    print("Testing /v1/models...")
    response = httpx.get(f"{BASE_URL}/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert len(data["data"]) > 0
    model = data["data"][0]
    assert "id" in model
    assert model["object"] == "model"
    print(f"  Model: {model['id']}")
    print("PASSED: test_models\n")


def test_chat_completions():
    """Test the /v1/chat/completions endpoint."""
    print("Testing /v1/chat/completions...")
    response = httpx.post(
        f"{BASE_URL}/v1/chat/completions",
        json={
            "model": "test",
            "messages": [
                {"role": "user", "content": "What is 2+2? Answer with just the number."}
            ],
            "max_tokens": 10,
        },
        timeout=60.0,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"
    assert data["id"].startswith("chatcmpl-")
    assert len(data["choices"]) == 1
    choice = data["choices"][0]
    assert choice["message"]["role"] == "assistant"
    assert choice["finish_reason"] in ["stop", "length"]
    assert "usage" in data
    print(f"  Response: {choice['message']['content']!r}")
    print(f"  Finish reason: {choice['finish_reason']}")
    print(f"  Usage: {data['usage']}")
    print("PASSED: test_chat_completions\n")


def test_chat_completions_with_logprobs():
    """Test chat completions with logprobs enabled."""
    print("Testing /v1/chat/completions with logprobs...")
    response = httpx.post(
        f"{BASE_URL}/v1/chat/completions",
        json={
            "model": "test",
            "messages": [
                {"role": "user", "content": "Say hello concisely and to the point."}
            ],
            "max_tokens": 10,
            "logprobs": True,
        },
        timeout=60.0,
    )
    assert response.status_code == 200
    data = response.json()
    choice = data["choices"][0]
    assert choice["logprobs"] is not None
    assert "content" in choice["logprobs"]
    logprobs = choice["logprobs"]["content"]
    assert len(logprobs) > 0
    print(f"  Response: {choice['message']['content']!r}")
    print(f"  Logprobs for first 3 tokens:")
    for lp in logprobs[:3]:
        print(f"    {lp['token']!r}: {lp['logprob']:.4f}")
    print("PASSED: test_chat_completions_with_logprobs\n")


def test_streaming_not_implemented():
    """Test that streaming returns 501 Not Implemented."""
    print("Testing streaming returns 501...")
    response = httpx.post(
        f"{BASE_URL}/v1/chat/completions",
        json={
            "model": "test",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        },
    )
    assert response.status_code == 501
    data = response.json()
    assert "error" in data
    assert data["error"]["type"] == "not_implemented"
    print("PASSED: test_streaming_not_implemented\n")


def test_openai_client():
    """Test using the official OpenAI Python client."""
    print("Testing with OpenAI client...")
    try:
        from openai import OpenAI
    except ImportError:
        print("SKIPPED: openai package not installed\n")
        return

    client = OpenAI(base_url=f"{BASE_URL}/v1", api_key="not-needed")

    # List models
    models = client.models.list()
    print(f"  Models: {[m.id for m in models.data]}")

    # Chat completion
    response = client.chat.completions.create(
        model="test",
        messages=[{"role": "user", "content": "Say hi concisely and to the point."}],
        max_tokens=1024,
    )
    print(f"  Response: {response.choices[0].message.content!r}")
    print("PASSED: test_openai_client\n")


if __name__ == "__main__":
    test_health()
    test_models()
    test_chat_completions()
    test_chat_completions_with_logprobs()
    test_streaming_not_implemented()
    test_openai_client()
    print("=" * 40)
    print("All API tests passed!")
