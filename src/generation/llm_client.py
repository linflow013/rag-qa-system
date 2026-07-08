"""LLM client — unified interface for DeepSeek API and local models."""

import logging
import os
import time
from dataclasses import dataclass
from typing import List, Optional

from config import config

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    model_name: str


@dataclass
class Message:
    role: str  # "user" or "assistant"
    content: str


class LLMClient:
    """Base class for LLM interactions."""

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        history: Optional[List[Message]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> GenerationResult:
        raise NotImplementedError


class DeepSeekAnthropicClient(LLMClient):
    """Uses Anthropic SDK to call DeepSeek's Anthropic-compatible API."""

    def __init__(self):
        self.api_key = os.environ.get(
            "ANTHROPIC_AUTH_TOKEN",
            os.environ.get("ANTHROPIC_API_KEY", ""),
        )
        self.base_url = os.environ.get(
            "ANTHROPIC_BASE_URL", "https://api.deepseek.com/anthropic"
        )
        self.model = os.environ.get("ANTHROPIC_MODEL", "deepseek-v4-pro[1m]")

        from anthropic import Anthropic
        self._client = Anthropic(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        history: Optional[List[Message]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> GenerationResult:
        temperature = temperature if temperature is not None else config.generation.temperature
        max_tokens = max_tokens or config.generation.max_tokens

        # Build messages in Anthropic format
        messages = []
        if history:
            for msg in history:
                role = "assistant" if msg.role == "assistant" else "user"
                messages.append({"role": role, "content": msg.content})
        messages.append({"role": "user", "content": prompt})

        start = time.time()
        try:
            kwargs = {
                "model": self.model,
                "max_tokens": max_tokens,
                "messages": messages,
            }
            if system_prompt:
                kwargs["system"] = system_prompt
            if temperature > 0:
                kwargs["temperature"] = temperature

            response = self._client.messages.create(**kwargs)

            latency = (time.time() - start) * 1000

            # Extract text from response (skip thinking blocks)
            text = ""
            if response.content:
                for block in response.content:
                    block_type = getattr(block, "type", "")
                    if block_type == "text" and hasattr(block, "text"):
                        text += block.text

            result = GenerationResult(
                text=text,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                latency_ms=latency,
                model_name=self.model,
            )
            logger.info(
                "LLM call: %d in / %d out tokens, %.0f ms",
                result.input_tokens, result.output_tokens, latency,
            )
            return result

        except Exception as e:
            latency = (time.time() - start) * 1000
            logger.error("LLM call failed after %.0f ms: %s", latency, e)
            return GenerationResult(
                text=f"[Error: LLM generation failed — {e}]",
                input_tokens=0,
                output_tokens=0,
                latency_ms=latency,
                model_name=self.model,
            )


class DeepSeekOpenAIClient(LLMClient):
    """OpenAI-compatible client for DeepSeek API (fallback)."""

    def __init__(self):
        self.api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        self.base_url = os.environ.get(
            "DEEPSEEK_BASE_URL", "https://api.deepseek.com"
        )
        self.model = "deepseek-chat"

        from openai import OpenAI
        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        history: Optional[List[Message]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> GenerationResult:
        temperature = temperature if temperature is not None else config.generation.temperature
        max_tokens = max_tokens or config.generation.max_tokens

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history:
            for msg in history:
                messages.append({"role": msg.role, "content": msg.content})
        messages.append({"role": "user", "content": prompt})

        start = time.time()
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            latency = (time.time() - start) * 1000

            result = GenerationResult(
                text=response.choices[0].message.content,
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                latency_ms=latency,
                model_name=self.model,
            )
            logger.info(
                "LLM call: %d in / %d out tokens, %.0f ms",
                result.input_tokens, result.output_tokens, latency,
            )
            return result

        except Exception as e:
            latency = (time.time() - start) * 1000
            logger.error("LLM call failed after %.0f ms: %s", latency, e)
            return GenerationResult(
                text=f"[Error: LLM generation failed — {e}]",
                input_tokens=0,
                output_tokens=0,
                latency_ms=latency,
                model_name=self.model,
            )


class MockLLMClient(LLMClient):
    """Mock client for testing without API access."""

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        history: Optional[List[Message]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> GenerationResult:
        return GenerationResult(
            text="[Mock LLM Response] This is a placeholder answer. "
                 "Set DEEPSEEK_API_KEY in .env to use the real API.",
            input_tokens=len(prompt) // 4,
            output_tokens=50,
            latency_ms=10.0,
            model_name="mock",
        )


def get_llm_client() -> LLMClient:
    """Factory: return appropriate LLM client based on available credentials."""

    # Priority 1: Anthropic-compatible API (DeepSeek via Anthropic SDK)
    anthropic_token = os.environ.get(
        "ANTHROPIC_AUTH_TOKEN",
        os.environ.get("ANTHROPIC_API_KEY", ""),
    )
    if anthropic_token and anthropic_token.startswith("sk-"):
        try:
            import anthropic
            logger.info("Using DeepSeek Anthropic-compatible client (model=%s)",
                        os.environ.get("ANTHROPIC_MODEL", "deepseek-v4-pro[1m]"))
            return DeepSeekAnthropicClient()
        except ImportError:
            logger.warning("anthropic SDK not installed; falling back...")

    # Priority 2: OpenAI-compatible API (DeepSeek native)
    deepseek_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if deepseek_key:
        try:
            return DeepSeekOpenAIClient()
        except Exception as e:
            logger.warning("Failed to init DeepSeek OpenAI client: %s", e)

    logger.warning("No LLM API key found. Using MockLLMClient.")
    return MockLLMClient()
