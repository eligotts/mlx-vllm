"""OpenAI-compatible API routes."""

import time
import uuid

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from mlx_vllm.engine.async_engine import AsyncEngine
from mlx_vllm.types import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ChoiceLogprobs,
    ModelInfo,
    ModelListResponse,
    TokenLogprob,
    Usage,
)

router = APIRouter(prefix="/v1")


def get_engine(request: Request) -> AsyncEngine:
    """Get the engine from app state."""
    return request.app.state.engine


@router.post("/chat/completions", response_model=None)
async def chat_completions(
    request: Request,
    body: ChatCompletionRequest,
) -> ChatCompletionResponse | JSONResponse:
    """Generate a chat completion."""
    engine = get_engine(request)

    if body.stream:
        # Streaming not yet implemented
        return JSONResponse(
            status_code=501,
            content={
                "error": {
                    "message": "Streaming not yet implemented",
                    "type": "not_implemented",
                    "code": None,
                }
            },
        )

    # Apply chat template and tokenize
    messages = [{"role": m.role, "content": m.content} for m in body.messages]
    prompt = engine.tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    prompt_tokens = engine.tokenizer.encode(prompt)

    # Submit to engine and wait for completion
    max_tokens = body.max_tokens or request.app.state.config.max_tokens
    result = await engine.generate(prompt_tokens, max_tokens=max_tokens)

    # Decode the response
    response_text = engine.tokenizer.decode(result.tokens)

    # Build logprobs if requested
    logprobs = None
    if body.logprobs:
        token_logprobs = []
        for token_id, logprob in zip(result.tokens, result.logprobs):
            token_str = engine.tokenizer.decode([token_id])
            token_logprobs.append(
                TokenLogprob(
                    token=token_str,
                    logprob=logprob,
                    bytes=list(token_str.encode("utf-8")),
                )
            )
        logprobs = ChoiceLogprobs(content=token_logprobs)

    # Build response
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    return ChatCompletionResponse(
        id=completion_id,
        created=int(time.time()),
        model=engine.model_id,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=response_text),
                finish_reason=result.finish_reason,
                logprobs=logprobs,
            )
        ],
        usage=Usage(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=len(result.tokens),
            total_tokens=result.prompt_tokens + len(result.tokens),
        ),
    )


@router.get("/models")
async def list_models(request: Request) -> ModelListResponse:
    """List available models."""
    engine = get_engine(request)
    return ModelListResponse(
        data=[
            ModelInfo(
                id=engine.model_id,
                created=request.app.state.model_loaded_at,
            )
        ]
    )


@router.get("/health")
async def health(request: Request) -> dict:
    """Health check endpoint."""
    engine = get_engine(request)
    return {
        "status": "healthy" if engine.is_ready else "not_ready",
        "model": engine.model_id,
        "pending_requests": engine.num_pending,
        "active_requests": engine.num_active,
    }
