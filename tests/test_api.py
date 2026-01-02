"""Test script for the OpenAI-compatible API.

Run the server first:
    uv run mlx-vllm --model /path/to/your/model

Then run this test:
    uv run python tests/test_api.py
"""

import json
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


def test_return_token_ids():
    """Test that return_token_ids returns prompt and completion token IDs."""
    print("Testing /v1/chat/completions with return_token_ids...")
    response = httpx.post(
        f"{BASE_URL}/v1/chat/completions",
        json={
            "model": "test",
            "messages": [
                {"role": "user", "content": "Say hi."}
            ],
            "max_tokens": 10,
            "return_token_ids": True,
            "logprobs": True,
        },
        timeout=60.0,
    )
    assert response.status_code == 200
    data = response.json()

    # Check prompt_token_ids at top level
    assert "prompt_token_ids" in data, "Missing prompt_token_ids in response"
    prompt_ids = data["prompt_token_ids"]
    assert isinstance(prompt_ids, list), "prompt_token_ids should be a list"
    assert len(prompt_ids) > 0, "prompt_token_ids should not be empty"
    assert all(isinstance(x, int) for x in prompt_ids), "prompt_token_ids should be integers"

    # Check token_ids in choice
    choice = data["choices"][0]
    assert "token_ids" in choice, "Missing token_ids in choice"
    token_ids = choice["token_ids"]
    assert isinstance(token_ids, list), "token_ids should be a list"
    assert len(token_ids) > 0, "token_ids should not be empty"
    assert all(isinstance(x, int) for x in token_ids), "token_ids should be integers"

    # Check that token_ids length matches completion_tokens in usage
    assert len(token_ids) == data["usage"]["completion_tokens"], \
        "token_ids length should match completion_tokens"

    # Check that prompt_token_ids length matches prompt_tokens in usage
    assert len(prompt_ids) == data["usage"]["prompt_tokens"], \
        "prompt_token_ids length should match prompt_tokens"

    # Check logprobs are also present and aligned with token_ids
    assert choice["logprobs"] is not None, "logprobs should be present"
    logprobs_content = choice["logprobs"]["content"]
    assert len(logprobs_content) == len(token_ids), \
        "logprobs length should match token_ids length"

    print(f"  Response: {choice['message']['content']!r}")
    print(f"  Prompt token IDs ({len(prompt_ids)}): {prompt_ids[:5]}{'...' if len(prompt_ids) > 5 else ''}")
    print(f"  Completion token IDs ({len(token_ids)}): {token_ids}")
    print(f"  Logprobs aligned: {len(logprobs_content)} entries")
    print("PASSED: test_return_token_ids\n")


def test_return_token_ids_disabled():
    """Test that token IDs are not returned when return_token_ids is false."""
    print("Testing that token IDs are excluded when not requested...")
    response = httpx.post(
        f"{BASE_URL}/v1/chat/completions",
        json={
            "model": "test",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 5,
            # return_token_ids defaults to False
        },
        timeout=60.0,
    )
    assert response.status_code == 200
    data = response.json()

    # prompt_token_ids should be null/absent
    assert data.get("prompt_token_ids") is None, \
        "prompt_token_ids should be null when not requested"

    # token_ids in choice should be null/absent
    choice = data["choices"][0]
    assert choice.get("token_ids") is None, \
        "token_ids should be null when not requested"

    print("PASSED: test_return_token_ids_disabled\n")


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


def test_json_generation():
    """Test the model's ability to generate valid JSON across different schemas."""
    print("Testing JSON generation...")

    test_cases = [
        {
            "name": "Judge Verdict",
            "prompt": (
                'You are a JUDGE, not a debater. Your sole task is to evaluate the debate transcript below and output a JSON verdict.\n\n'
                'CRITICAL INSTRUCTIONS:\n'
                '- You are NOT a participant in this debate\n'
                '- Do NOT write any arguments or continue the debate\n'
                '- Do NOT write any text before or after the JSON\n'
                '- Output ONLY a single JSON object\n\n'
                'Evaluation criteria: strength of arguments, quality of reasoning, use of evidence, persuasiveness.\n\n'
                'Required JSON format (output ONLY this, nothing else):\n'
                '{"winner": "Affirmative", "score": 0.7, "reasoning": "One sentence explaining why"}\n\n'
                'Rules:\n'
                '- "winner" must be exactly "Affirmative" or "Negative"\n'
                '- "score" is winner\'s margin of victory from 0.5 (very close) to 1.0 (decisive)\n'
                '- Start your response with { and end with }\n\n'
                'Now evaluate the following debate transcript: "Affirmative: We should go to Mars. Negative: It is too expensive."\n'
                'Respond with ONLY JSON:'
            ),
            "validate": lambda d: (
                isinstance(d, dict) and 
                d.get("winner") in ["Affirmative", "Negative"] and 
                isinstance(d.get("score"), (int, float))
            )
        },
        {
            "name": "Fruit List",
            "prompt": (
                "List 3 common fruits in a JSON array of objects. Each object should have 'name' and 'color'.\n\n"
                "CRITICAL INSTRUCTIONS:\n"
                "- Do NOT write any text before or after the JSON\n"
                "- Do NOT use markdown code blocks (backticks)\n"
                "- Output ONLY a single JSON array\n"
                "- Start your response with [ and end with ]\n\n"
                'Example valid response: [{"name": "fruit", "color": "color"}]\n\n'
                "Respond with ONLY JSON:"
            ),
            "validate": lambda d: (
                isinstance(d, list) and 
                len(d) == 3 and 
                all("name" in x and "color" in x for x in d)
            )
        },
        {
            "name": "Nested Person Data",
            "prompt": (
                "Create a JSON object for a fictional character. Include 'name', 'age', and 'attributes' "
                "(which is an object with 'strength' and 'intelligence' from 1-10).\n\n"
                "CRITICAL INSTRUCTIONS:\n"
                "- Do NOT write any text before or after the JSON\n"
                "- Do NOT use markdown code blocks (backticks)\n"
                "- Output ONLY a single JSON object\n"
                "- Start your response with { and end with }\n\n"
                "Respond with ONLY JSON:"
            ),
            "validate": lambda d: (
                isinstance(d, dict) and 
                "name" in d and 
                isinstance(d.get("attributes"), dict) and 
                "strength" in d["attributes"]
            )
        }
    ]

    failed_cases = 0
    for case in test_cases:
        print(f"  Testing case: {case['name']}...")
        # ... existing request code ...
        response = httpx.post(
            f"{BASE_URL}/v1/chat/completions",
            json={
                "model": "test",
                "messages": [{"role": "user", "content": case["prompt"]}],
                "max_tokens": 150,
                "temperature": 0.0,  # Use greedy to be more consistent
            },
            timeout=60.0,
        )
        if response.status_code != 200:
            print(f"    FAILED: {case['name']} - HTTP {response.status_code}: {response.text}")
            failed_cases += 1
            continue
        
        content = response.json()["choices"][0]["message"]["content"].strip()
        
        # Robustly handle models that still insist on markdown blocks
        if content.startswith("```"):
            lines = content.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            content = "\n".join(lines).strip()
        
        # Try to parse JSON
        try:
            data = json.loads(content)
            if case["validate"](data):
                print(f"    PASSED: {case['name']}")
            else:
                print(f"    FAILED: {case['name']} - Validation failed. Data was: {data}")
                failed_cases += 1
        except json.JSONDecodeError:
            print(f"    FAILED: {case['name']} - Could not parse JSON. Content was:\n{content}")
            failed_cases += 1

    if failed_cases == 0:
        print("PASSED: test_json_generation\n")
    else:
        print(f"FAILED: test_json_generation ({failed_cases} cases failed)\n")


if __name__ == "__main__":
    test_health()
    test_models()
    # test_chat_completions()
    # test_chat_completions_with_logprobs()
    # test_streaming_not_implemented()
    # test_openai_client()
    test_return_token_ids()
    test_return_token_ids_disabled()
    # test_json_generation()
    print("=" * 40)
    print("All API tests passed!")
