"""
vLLM Client for Speculative RAG.
Provides OpenAI-compatible interface for local vLLM server.
"""

from __future__ import annotations

import json
import re
from types import SimpleNamespace
from typing import Any

import httpx
from loguru import logger
from pydantic import BaseModel


class VLLMClient:
    """Client for vLLM server that mimics OpenAI API interface."""

    def __init__(self, base_url: str = "http://localhost:8010", model_name: str = "Qwen3-32B"):
        self.base_url = base_url
        self.model_name = model_name
        # Increase timeout and add retry logic for concurrent requests
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(180.0, read=180.0, write=180.0, connect=10.0),
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )

        # Create nested structure to mimic OpenAI client
        self.beta = self._Beta(self)
        self.chat = self._Chat(self)

    class _Beta:
        def __init__(self, parent: VLLMClient):
            self.parent = parent
            self.chat = self._BetaChat(parent)

        class _BetaChat:
            def __init__(self, parent: VLLMClient):
                self.parent = parent
                self.completions = self._Completions(parent)

            class _Completions:
                def __init__(self, parent: VLLMClient):
                    self.parent = parent

                async def parse(
                    self,
                    model: str,
                    messages: list[dict[str, str]],
                    response_format: type[BaseModel],
                    temperature: float = 0.0,
                    logprobs: bool = True,
                    max_tokens: int = 512,
                    **kwargs: Any,
                ) -> Any:
                    """Parse structured output from vLLM (mimics OpenAI's beta.chat.completions.parse)."""
                    # Extract system message as prompt
                    prompt = messages[0]["content"] if messages else ""

                    # Add JSON format instruction
                    json_prompt = f"""{prompt}

Respond ONLY with valid JSON in this exact format:
{{
    "rationale": "your reasoning here",
    "response": "your answer here"
}}"""

                    # Call vLLM
                    url = f"{self.parent.base_url}/generate/"
                    params = {
                        "prompt": json_prompt,
                        "max_length": max_tokens,
                        "temperature": temperature,
                        "do_sample": temperature > 0,
                    }

                    response = await self.parent.client.get(url, params=params)
                    response.raise_for_status()
                    data = response.json()

                    generated_text = data.get("generated_texts", [""])[0]

                    # Try to parse JSON from the response
                    try:
                        # Extract JSON from markdown code blocks if present
                        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", generated_text, re.DOTALL)
                        if json_match:
                            json_str = json_match.group(1)
                        else:
                            # Try to find JSON object directly
                            json_match = re.search(r"\{.*\}", generated_text, re.DOTALL)
                            json_str = json_match.group(0) if json_match else generated_text

                        parsed_data = json.loads(json_str)
                        parsed_obj = response_format(**parsed_data)
                    except Exception as e:
                        logger.warning(f"Failed to parse JSON from vLLM response: {e}")
                        logger.debug(f"Raw response: {generated_text}")
                        # Fallback: create object with raw text
                        parsed_obj = response_format(
                            rationale="Unable to parse structured response",
                            response=generated_text[:200],
                        )

                    # Create mock logprobs (vLLM doesn't provide detailed logprobs in this setup)
                    num_tokens = data.get("output_token_counts", [10])[0]
                    mock_logprobs = [SimpleNamespace(logprob=-0.1) for _ in range(num_tokens)]

                    choice = SimpleNamespace(
                        message=SimpleNamespace(parsed=parsed_obj),
                        logprobs=SimpleNamespace(content=mock_logprobs),
                    )

                    return SimpleNamespace(choices=[choice])

    class _Chat:
        def __init__(self, parent: VLLMClient):
            self.parent = parent
            self.completions = self._Completions(parent)

        class _Completions:
            def __init__(self, parent: VLLMClient):
                self.parent = parent

            async def create(
                self,
                model: str,
                messages: list[dict[str, str]],
                temperature: float = 0.0,
                logprobs: bool = True,
                max_tokens: int = 2,
                **kwargs: Any,
            ) -> Any:
                """Create chat completion (mimics OpenAI's chat.completions.create)."""
                # Extract system message as prompt
                prompt = messages[0]["content"] if messages else ""

                # Call vLLM
                url = f"{self.parent.base_url}/generate/"
                params = {
                    "prompt": prompt,
                    "max_length": max_tokens,
                    "temperature": temperature,
                    "do_sample": temperature > 0,
                }

                response = await self.parent.client.get(url, params=params)
                response.raise_for_status()
                data = response.json()

                generated_text = data.get("generated_texts", [""])[0].strip()

                # Create mock logprobs
                num_tokens = data.get("output_token_counts", [1])[0]
                mock_logprobs = [SimpleNamespace(logprob=-0.05) for _ in range(num_tokens)]

                choice = SimpleNamespace(
                    message=SimpleNamespace(content=generated_text),
                    logprobs=SimpleNamespace(content=mock_logprobs),
                )

                return SimpleNamespace(choices=[choice])

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


__all__ = ["VLLMClient"]
