"""Test script for ContinuousBatchingEngine."""

from mlx_lm import load

from mlx_vllm import ContinuousBatchingEngine

MODEL_PATH = "/Users/eligottlieb/.lmstudio/models/lmstudio-community/Qwen2.5-7B-Instruct-MLX-4bit"


def get_extra_stop_tokens(tokenizer) -> set[int]:
    """Get additional stop tokens beyond eos_token_ids (e.g., <|endoftext|>)."""
    extra = set()
    for token in ["<|endoftext|>", "<|eot_id|>"]:
        ids = tokenizer.encode(token, add_special_tokens=False)
        if len(ids) == 1:
            extra.add(ids[0])
    return extra - set(tokenizer.eos_token_ids)


def test_basic_generation():
    """Test basic single-prompt generation."""
    print("Loading model...")
    model, tokenizer = load(MODEL_PATH)

    extra_stop = get_extra_stop_tokens(tokenizer)
    print(f"Extra stop tokens: {extra_stop}")

    engine = ContinuousBatchingEngine(
        model,
        tokenizer,
        max_batch_size=8,
        default_max_tokens=50,
        extra_stop_tokens=extra_stop,
    )

    prompt = "What is 2 + 2? BE CONCISE AND TO THE POINT."
    prompt_tokens = tokenizer.encode(prompt)
    print(f"Prompt: {prompt!r}")
    print(f"Prompt tokens: {len(prompt_tokens)}")

    request_id = engine.add(prompt_tokens, max_tokens=50)
    print(f"Added request {request_id}")

    outputs = []
    steps = 0
    while engine.has_work():
        completed = engine.step()
        outputs.extend(completed)
        steps += 1

    print(f"Completed in {steps} steps")
    assert len(outputs) == 1
    output = outputs[0]

    print(f"Request ID: {output.request_id}")
    print(f"Generated tokens: {len(output.tokens)}")
    print(f"Logprobs: {len(output.logprobs)}")
    print(f"Finish reason: {output.finish_reason}")
    assert len(output.tokens) == len(output.logprobs), "tokens and logprobs must align"

    decoded = tokenizer.decode(output.tokens)
    print(f"Response: {decoded!r}")
    print(f"Sum of logprobs: {sum(output.logprobs):.4f}")

    engine.close()
    print("PASSED: test_basic_generation\n")


def test_batch_generation():
    """Test multiple prompts processed as a batch."""
    print("Loading model...")
    model, tokenizer = load(MODEL_PATH)

    engine = ContinuousBatchingEngine(
        model,
        tokenizer,
        max_batch_size=8,
        default_max_tokens=1024,
        extra_stop_tokens=get_extra_stop_tokens(tokenizer),
    )

    prompts = [
        "Say hello in French. BE CONCISE AND TO THE POINT.",
        "Say hello in Spanish. BE CONCISE AND TO THE POINT.",
        "Say hello in German. BE CONCISE AND TO THE POINT.",
        "Say hello in Japanese. BE CONCISE AND TO THE POINT.",
    ]

    request_ids = []
    for prompt in prompts:
        prompt_tokens = tokenizer.encode(prompt)
        rid = engine.add(prompt_tokens, max_tokens=1024)
        request_ids.append(rid)
        print(f"Added request {rid}: {prompt!r}")

    outputs = []
    steps = 0
    while engine.has_work():
        completed = engine.step()
        outputs.extend(completed)
        steps += 1
        if completed:
            print(f"Step {steps}: {len(completed)} request(s) completed")

    print(f"Total steps: {steps}")
    assert len(outputs) == len(prompts), f"Expected {len(prompts)} outputs, got {len(outputs)}"

    # Sort by request_id to match original order
    outputs.sort(key=lambda x: x.request_id)

    for prompt, output in zip(prompts, outputs):
        assert len(output.tokens) == len(output.logprobs)
        decoded = tokenizer.decode(output.tokens)
        print(f"[{output.request_id}] {prompt!r} -> {decoded!r}")
        print(f"    tokens={len(output.tokens)}, logprob_sum={sum(output.logprobs):.4f}, finish={output.finish_reason}")

    engine.close()
    print("PASSED: test_batch_generation\n")


def test_continuous_insertion():
    """Test adding new requests while others are generating."""
    print("Loading model...")
    model, tokenizer = load(MODEL_PATH)

    engine = ContinuousBatchingEngine(
        model,
        tokenizer,
        max_batch_size=4,
        default_max_tokens=1024,
        extra_stop_tokens=get_extra_stop_tokens(tokenizer),
    )

    # Add initial batch
    initial_prompts = ["Count to 3. BE CONCISE AND TO THE POINT.", "Count to 5. BE CONCISE AND TO THE POINT."]
    for prompt in initial_prompts:
        engine.add(tokenizer.encode(prompt), max_tokens=1024)
    print(f"Added {len(initial_prompts)} initial prompts")

    outputs = []
    steps = 0
    added_mid_generation = False

    while engine.has_work():
        completed = engine.step()
        outputs.extend(completed)
        steps += 1

        # Add more requests mid-generation
        if steps == 5 and not added_mid_generation:
            new_prompts = ["What is 1+1? BE CONCISE AND TO THE POINT.", "What is 2+2? BE CONCISE AND TO THE POINT."]
            for prompt in new_prompts:
                engine.add(tokenizer.encode(prompt), max_tokens=1024)
            print(f"Step {steps}: Added {len(new_prompts)} more prompts mid-generation")
            added_mid_generation = True

    print(f"Total outputs: {len(outputs)}")
    print(f"Total steps: {steps}")
    assert len(outputs) == 4, f"Expected 4 outputs, got {len(outputs)}"

    for output in outputs:
        assert len(output.tokens) == len(output.logprobs)
        decoded = tokenizer.decode(output.tokens)
        print(f"[{output.request_id}] {decoded!r} (finish={output.finish_reason})")

    engine.close()
    print("PASSED: test_continuous_insertion\n")


def test_varying_lengths():
    """Test that requests with different max_tokens work correctly."""
    print("Loading model...")
    model, tokenizer = load(MODEL_PATH)

    engine = ContinuousBatchingEngine(
        model,
        tokenizer,
        max_batch_size=8,
        default_max_tokens=1024,
        extra_stop_tokens=get_extra_stop_tokens(tokenizer),
    )

    # Different max_tokens for each
    configs = [
        ("Short response: BE CONCISE AND TO THE POINT.", 10),
        ("Medium response: BE CONCISE AND TO THE POINT.", 30),
        ("Longer response: BE CONCISE AND TO THE POINT.", 50),
    ]

    for prompt, max_tokens in configs:
        engine.add(tokenizer.encode(prompt), max_tokens=max_tokens)
        print(f"Added: {prompt!r} with max_tokens={max_tokens}")

    outputs = []
    while engine.has_work():
        outputs.extend(engine.step())

    outputs.sort(key=lambda x: x.request_id)

    for (prompt, max_tokens), output in zip(configs, outputs):
        assert len(output.tokens) == len(output.logprobs)
        assert len(output.tokens) <= max_tokens
        decoded = tokenizer.decode(output.tokens)
        print(f"[{output.request_id}] max={max_tokens}, got={len(output.tokens)}, finish={output.finish_reason}")
        print(f"    {decoded!r}")

    engine.close()
    print("PASSED: test_varying_lengths\n")


if __name__ == "__main__":
    test_basic_generation()
    test_batch_generation()
    test_continuous_insertion()
    # test_varying_lengths()
    print("=" * 40)
    print("All tests passed!")
