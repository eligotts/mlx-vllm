"""Async wrapper for the continuous batching engine."""

import asyncio
from dataclasses import dataclass

from mlx_vllm.engine.generation import ContinuousBatchingEngine


@dataclass
class AsyncGenerationResult:
    """Result from async generation."""

    request_id: int
    tokens: list[int]
    logprobs: list[float]
    finish_reason: str
    prompt_tokens: int


class AsyncEngine:
    """
    Async wrapper around ContinuousBatchingEngine.

    Runs the engine loop in a background task and provides async interface
    for submitting generation requests.
    """

    def __init__(
        self,
        model,
        tokenizer,
        max_batch_size: int = 32,
        default_max_tokens: int = 4096,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.model_id: str = getattr(tokenizer, "name_or_path", "unknown")

        # Build stop tokens
        extra_stop = self._get_extra_stop_tokens(tokenizer)

        self._engine = ContinuousBatchingEngine(
            model=model,
            tokenizer=tokenizer,
            max_batch_size=max_batch_size,
            default_max_tokens=default_max_tokens,
            extra_stop_tokens=extra_stop,
        )

        # request_id -> (Future, prompt_token_count)
        self._pending_futures: dict[int, tuple[asyncio.Future[AsyncGenerationResult], int]] = {}
        self._loop_task: asyncio.Task | None = None
        self._running = False

    def _get_extra_stop_tokens(self, tokenizer) -> set[int]:
        """Get additional stop tokens beyond eos_token_ids."""
        extra = set()
        for token in ["<|endoftext|>", "<|eot_id|>", "<|end|>"]:
            ids = tokenizer.encode(token, add_special_tokens=False)
            if len(ids) == 1:
                extra.add(ids[0])
        return extra - set(tokenizer.eos_token_ids)

    async def start(self) -> None:
        """Start the background engine loop."""
        if self._running:
            return
        self._running = True
        self._loop_task = asyncio.create_task(self._engine_loop())

    async def stop(self) -> None:
        """Stop the background engine loop."""
        self._running = False
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
        self._engine.close()

    async def _engine_loop(self) -> None:
        """Background loop that runs engine steps."""
        while self._running:
            if not self._engine.has_work():
                # No work to do, yield control and wait a bit
                await asyncio.sleep(0.001)
                continue

            # Run one step (this is blocking but should be fast)
            completed = self._engine.step()

            # Resolve futures for completed requests
            for output in completed:
                if output.request_id in self._pending_futures:
                    future, prompt_tokens = self._pending_futures.pop(output.request_id)
                    result = AsyncGenerationResult(
                        request_id=output.request_id,
                        tokens=output.tokens,
                        logprobs=output.logprobs,
                        finish_reason=output.finish_reason,
                        prompt_tokens=prompt_tokens,
                    )
                    if not future.done():
                        future.set_result(result)

            # Yield control to allow other coroutines to run
            await asyncio.sleep(0)

    async def generate(
        self,
        prompt_tokens: list[int],
        max_tokens: int | None = None,
    ) -> AsyncGenerationResult:
        """
        Submit a generation request and wait for completion.

        Args:
            prompt_tokens: Tokenized prompt
            max_tokens: Maximum tokens to generate

        Returns:
            AsyncGenerationResult with tokens, logprobs, and finish reason
        """
        loop = asyncio.get_running_loop()
        future: asyncio.Future[AsyncGenerationResult] = loop.create_future()

        request_id = self._engine.add(prompt_tokens, max_tokens)
        self._pending_futures[request_id] = (future, len(prompt_tokens))

        return await future

    @property
    def num_pending(self) -> int:
        """Number of requests waiting to enter the batch."""
        return self._engine.num_pending

    @property
    def num_active(self) -> int:
        """Number of requests currently generating."""
        return self._engine.num_active

    @property
    def is_ready(self) -> bool:
        """Check if engine is ready to accept requests."""
        return self._running
