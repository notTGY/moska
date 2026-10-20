import asyncio
import os
import signal
import socket
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from typing import Any, Literal

import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatResult
from langchain_openai import ChatOpenAI
from langchain_openrouter import ChatOpenRouter
from pydantic import BaseModel, Field


_QWEN_3_5_4B_PORT = 8024
_QWEN_3_8_FLASH_NEXT_PORT = 8025


class ChatVLLM(ChatOpenAI):
    """ChatOpenAI adapter for vLLM's Chat Completions reasoning field."""

    preserve_reasoning: bool = Field(default=False, exclude=True)

    def _create_chat_result(
        self,
        response: dict[str, Any] | BaseModel,
        generation_info: dict[str, Any] | None = None,
    ) -> ChatResult:
        result = super()._create_chat_result(response, generation_info)
        response_dict = (
            response
            if isinstance(response, dict)
            else response.model_dump(warnings=False)
        )

        for choice in response_dict.get("choices") or []:
            if choice.get("finish_reason") == "length":
                raise RuntimeError(
                    "vLLM exhausted max_completion_tokens before completing "
                    "its response."
                )

        if not self.preserve_reasoning:
            return result

        for generation, choice in zip(
            result.generations,
            response_dict.get("choices") or [],
            strict=True,
        ):
            message = generation.message
            response_message = choice.get("message") or {}
            reasoning = response_message.get("reasoning")
            if reasoning is None:
                reasoning = response_message.get("reasoning_content")

            if (
                isinstance(message, AIMessage)
                and isinstance(reasoning, str)
                and reasoning
            ):
                message.additional_kwargs["reasoning_content"] = reasoning

        return result

    def _get_request_payload(
        self,
        input_: Any,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        request_messages = payload.get("messages")
        if not isinstance(request_messages, list):
            raise RuntimeError("ChatVLLM requires the Chat Completions API.")

        if not self.preserve_reasoning:
            return payload

        messages = self._convert_input(input_).to_messages()
        for message, request_message in zip(
            messages,
            request_messages,
            strict=True,
        ):
            reasoning = message.additional_kwargs.get("reasoning_content")
            if reasoning is None:
                reasoning = message.additional_kwargs.get("reasoning")
            if (
                isinstance(message, AIMessage)
                and isinstance(reasoning, str)
                and reasoning
            ):
                request_message["reasoning"] = reasoning

        return payload


def create_model(
    model_id: Literal[
        "vllm/qwen_3_5_4b_thinking",
        "vllm/qwen_3_5_4b_non_thinking",
        "vllm/qwen_3_8_flash_next_non_thinking",
        "lmrouter/qwen_3_5_4b_unlooped_thinking",
        "lmrouter/qwen_3_5_4b_unlooped_non_thinking",
        "lmrouter/qwen_3_8_flash_next_non_thinking",
        "lmrouter/qwen_3_8_flash_next_low_thinking",
        "lmrouter/qwen_3_8_flash_next_medium_thinking",
        "temp/qwen_3_5_4b_unlooped_thinking",
        "temp/qwen_3_8_flash_next_non_thinking",
        "openrouter/qwen_3_8_flash_next_non_thinking",
        "openrouter/qwen_3_8_flash_next_low_thinking",
        "openrouter/qwen_3_8_flash_next_medium_thinking",
    ],
    *,
    base_url: str | None = None,
) -> BaseChatModel:
    if base_url is not None and not model_id.startswith("vllm/"):
        raise ValueError("base_url is supported only for vllm models")
    if not model_id.startswith("openrouter/"):
        socket = os.environ.get("LLM_PROXY_UNIX_SOCKET") if model_id.startswith("lmrouter/") else None
        sync_client = httpx.Client(
            transport=httpx.HTTPTransport(uds=socket) if socket else None,
            trust_env=False,
        )
        client = httpx.AsyncClient(
            transport=httpx.AsyncHTTPTransport(uds=socket) if socket else None,
            trust_env=False,
        )
    match model_id:
        case "vllm/qwen_3_5_4b_thinking" | "vllm/qwen_3_5_4b_non_thinking":
            thinking = model_id == "vllm/qwen_3_5_4b_thinking"
            return ChatVLLM(
                model="Qwen/Qwen3.5-4B",
                base_url=base_url or f"http://{os.environ.get('VLLM_HOST', '127.0.0.1')}:{_QWEN_3_5_4B_PORT}/v1",
                api_key="EMPTY",
                # https://huggingface.co/Qwen/Qwen3.5-4B#best-practices
                temperature=1.0 if thinking else 0.7,
                top_p=0.95 if thinking else 0.8,
                presence_penalty=1.5,
                extra_body={
                    "top_k": 20,
                    "min_p": 0.0,
                    "repetition_penalty": 1.0,
                    "chat_template_kwargs": {"enable_thinking": thinking},
                },
                preserve_reasoning=thinking,
                timeout=600,
                max_retries=2,
                http_client=sync_client,
                http_async_client=client,
                disable_streaming=True,
                use_responses_api=False,
            )
        case "vllm/qwen_3_8_flash_next_non_thinking":
            return ChatVLLM(
                model="Qwen/Qwen3.8-Flash-Next-FP8",
                base_url=base_url or f"http://{os.environ.get('VLLM_HOST', '127.0.0.1')}:{_QWEN_3_8_FLASH_NEXT_PORT}/v1",
                api_key="EMPTY",
                temperature=0.7,
                top_p=0.8,
                presence_penalty=1.5,
                extra_body={
                    "top_k": 20,
                    "min_p": 0.0,
                    "repetition_penalty": 1.0,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
                preserve_reasoning=False,
                timeout=600,
                max_retries=2,
                http_client=sync_client,
                http_async_client=client,
                disable_streaming=True,
                use_responses_api=False,
            )
        case "lmrouter/qwen_3_5_4b_unlooped_thinking":
            return ChatVLLM(
                model="Qwen/Qwen3.5-4B-unlooped",
                base_url="https://lmrouter.2a2i.org/v1",
                api_key=os.environ.get("LLM_PROXY_MASTER_KEY", "EMPTY"),
                temperature=0.6,
                top_p=0.95,
                extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": True}},
                preserve_reasoning=True,
                timeout=600,
                max_retries=2,
                http_client=sync_client,
                http_async_client=client,
                disable_streaming=True,
                use_responses_api=False,
            )
        case "lmrouter/qwen_3_5_4b_unlooped_non_thinking":
            return ChatVLLM(
                model="Qwen/Qwen3.5-4B-unlooped",
                base_url="https://lmrouter.2a2i.org/v1",
                api_key=os.environ.get("LLM_PROXY_MASTER_KEY", "EMPTY"),
                temperature=0.7,
                top_p=0.8,
                extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}},
                preserve_reasoning=False,
                timeout=600,
                max_retries=2,
                http_client=sync_client,
                http_async_client=client,
                disable_streaming=True,
                use_responses_api=False,
            )
        case "lmrouter/qwen_3_8_flash_next_non_thinking":
            return ChatVLLM(
                model="Qwen/Qwen3.8-Flash-Next-NVFP4",
                base_url="https://lmrouter.2a2i.org/v1",
                api_key=os.environ.get("LLM_PROXY_MASTER_KEY", "EMPTY"),
                # https://huggingface.co/Qwen/Qwen3.8-Flash-Next#api-usage
                temperature=0.7,
                top_p=0.8,
                presence_penalty=1.5,
                extra_body={
                    "top_k": 20,
                    "min_p": 0.0,
                    "repetition_penalty": 1.0,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
                preserve_reasoning=False,
                timeout=600,
                max_retries=2,
                http_client=sync_client,
                http_async_client=client,
                disable_streaming=True,
                use_responses_api=False,
            )
        case "lmrouter/qwen_3_8_flash_next_low_thinking" | "lmrouter/qwen_3_8_flash_next_medium_thinking":
            return ChatVLLM(
                model="Qwen/Qwen3.8-Flash-Next-NVFP4",
                base_url="https://lmrouter.2a2i.org/v1",
                api_key=os.environ.get("LLM_PROXY_MASTER_KEY", "EMPTY"),
                # https://huggingface.co/Qwen/Qwen3.8-Flash-Next#api-usage
                temperature=1.0,
                top_p=0.95,
                presence_penalty=0.0,
                reasoning_effort=(
                    "low" if model_id == "lmrouter/qwen_3_8_flash_next_low_thinking" else "medium"
                ),
                extra_body={
                    "allowed_openai_params": ["reasoning_effort"],
                    "top_k": 20,
                    "min_p": 0.0,
                    "repetition_penalty": 1.0,
                    "chat_template_kwargs": {
                        "enable_thinking": True,
                        "preserve_thinking": True,
                    },
                },
                preserve_reasoning=True,
                timeout=600,
                max_retries=2,
                http_client=sync_client,
                http_async_client=client,
                disable_streaming=True,
                use_responses_api=False,
            )
        case "temp/qwen_3_5_4b_unlooped_thinking":
            return ChatVLLM(
                model="Qwen/Qwen3.5-4B-unlooped",
                base_url="https://135-106-142-250.sslip.io/v1",
                api_key=os.environ.get("TEMP_LLM_PROXY_MASTER_KEY", "EMPTY"),
                temperature=0.6,
                top_p=0.95,
                extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": True}},
                preserve_reasoning=True,
                timeout=600,
                max_retries=2,
                http_client=sync_client,
                http_async_client=client,
                disable_streaming=True,
                use_responses_api=False,
            )
        case "temp/qwen_3_8_flash_next_non_thinking":
            return ChatVLLM(
                model="Qwen/Qwen3.8-Flash-Next-FP8",
                base_url="https://135-106-142-250.sslip.io/v1",
                api_key=os.environ.get("TEMP_LLM_PROXY_MASTER_KEY", "EMPTY"),
                # https://huggingface.co/Qwen/Qwen3.8-Flash-Next#api-usage
                temperature=0.7,
                top_p=0.8,
                presence_penalty=1.5,
                extra_body={
                    "top_k": 20,
                    "min_p": 0.0,
                    "repetition_penalty": 1.0,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
                preserve_reasoning=False,
                timeout=600,
                max_retries=2,
                http_client=sync_client,
                http_async_client=client,
                disable_streaming=True,
                use_responses_api=False,
            )
        case "openrouter/qwen_3_8_flash_next_non_thinking" | "openrouter/qwen_3_8_flash_next_low_thinking" | "openrouter/qwen_3_8_flash_next_medium_thinking":
            thinking = model_id != "openrouter/qwen_3_8_flash_next_non_thinking"
            return ChatOpenRouter(
                model="qwen/qwen3.8-flash:nitro",
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ["OPENROUTER_API_KEY"],
                # https://huggingface.co/Qwen/Qwen3.8-Flash-Next#best-practices
                temperature=1.0 if thinking else 0.7,
                top_p=0.95 if thinking else 0.8,
                presence_penalty=0.0 if thinking else 1.5,
                model_kwargs={"top_k": 20},
                reasoning={
                    "effort": "low" if model_id == "openrouter/qwen_3_8_flash_next_low_thinking" else "medium",
                } if thinking else {"enabled": False},
                timeout=600_000,
                max_retries=2,
                disable_streaming=True,
            )
        case _:
            raise ValueError(f"Unknown model: {model_id}")


@asynccontextmanager
async def vllm_server(
    model_id: str, *, gpu: str, host: str = "127.0.0.1",
) -> AsyncIterator[None]:
    """Run a supported local model on one GPU and stop it on context exit."""
    match model_id:
        case "vllm/qwen_3_5_4b_thinking" | "vllm/qwen_3_5_4b_non_thinking":
            model = "Qwen/Qwen3.5-4B"
            port = _QWEN_3_5_4B_PORT
            command = [
                sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", model,
                "--host", host, "--port", str(port),
                "--max-model-len", "131072",
                "--gpu-memory-utilization", "0.90",
                "--language-model-only",
                "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder",
                "--reasoning-parser", "qwen3",
                "--enable-prefix-caching",
            ]
        case _:
            raise ValueError(f"Unsupported local vLLM model: {model_id}")

    client_host = "127.0.0.1" if host == "0.0.0.0" else host
    with socket.socket() as listener:
        listener.bind((host, port))
    process = await asyncio.create_subprocess_exec(
        *command, env=os.environ | {"CUDA_VISIBLE_DEVICES": gpu}, start_new_session=True,
    )
    try:
        async with httpx.AsyncClient(trust_env=False) as client:
            async with asyncio.timeout(600):
                while True:
                    if process.returncode is not None:
                        raise RuntimeError(f"Server exited during startup: {process.returncode}")
                    try:
                        response = await client.get(f"http://{client_host}:{port}/health", timeout=2)
                        if response.status_code == 200:
                            break
                    except httpx.TransportError:
                        pass
                    await asyncio.sleep(0.2)
        yield
    finally:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        try:
            await asyncio.wait_for(process.wait(), timeout=30)
        except TimeoutError:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            await process.wait()
