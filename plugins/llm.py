"""
LLM API Integration for Text Completion
@zhs LLM API 集成 - 文本补全
"""
import json
import logging
from typing import Any, Dict, List, Optional

import httpx

from jindai.pipeline import PipelineStage
from jindai.plugin import Plugin
from jindai.models import Paragraph


class LLMStageBase(PipelineStage):
    """Base class for LLM API integration stages.
    
    Provides common functionality for calling LLM APIs including:
    - HTTP client management
    - Request/response logging
    - Error handling
    """

    def __init__(self, name: str = "") -> None:
        """Initialize LLM stage.
        
        Args:
            name: Instance name for logging (default: empty string).
        """
        super().__init__(name)
        self.client = None
        self._initialized = False

    async def _ensure_client(self) -> httpx.AsyncClient:
        """Ensure HTTP client is initialized.
        
        Returns:
            Initialized httpx.AsyncClient instance.
        """
        if not self._initialized:
            self.client = httpx.AsyncClient(timeout=60.0)
            self._initialized = True
        return self.client

    async def _close_client(self) -> None:
        """Close HTTP client."""
        if self.client and self._initialized:
            await self.client.aclose()
            self._initialized = False

    def _log_request(self, messages: List[Dict[str, str]], model: str) -> None:
        """Log request details.
        
        Args:
            messages: List of message dictionaries.
            model: Model name used for the request.
        """
        if self.verbose:
            self._log(f"[LLM] Calling {model} with {len(messages)} messages")

    def _log_response(self, content: str, model: str) -> None:
        """Log response details.
        
        Args:
            content: Response content.
            model: Model name used for the request.
        """
        if self.verbose:
            self._log(f"[LLM] {model} response: {content[:100]}...")


class OpenAILikeCompletion(LLMStageBase):
    """Text completion using OpenAI-compatible API.
    
    Supports any API that follows the OpenAI API format including:
    - OpenAI (GPT-3.5, GPT-4, etc.)
    - Local LLM servers with OpenAI compatibility
    - Cloud providers with OpenAI-compatible endpoints
    """

    def __init__(
        self,
        api_url: str = "http://localhost:11434/v1/chat/completions",
        api_key: str = "",
        model: str = "gpt-3.5-turbo",
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        result_field: str = "llm_completion",
    ) -> None:
        """Initialize OpenAI-compatible LLM stage.
        
        Args:
            api_url: API endpoint URL (default: Ollama default)
            api_key: API key for authentication (default: empty)
            model: Model name to use (default: gpt-3.5-turbo)
            system_prompt: System prompt to set model behavior (default: empty)
            temperature: Sampling temperature (0.0-1.0, default: 0.7)
            max_tokens: Maximum tokens in response (default: 1000)
            result_field: Field name to store completion result (default: llm_completion)
        """
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.result_field = result_field
        super().__init__()

    async def resolve(self, paragraph: Paragraph) -> Paragraph:
        """Process paragraph through LLM API.
        
        Args:
            paragraph: Input paragraph to process.
            
        Returns:
            Paragraph with completion result added.
        """
        client = await self._ensure_client()
        
        # Build messages
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": paragraph.content})
        
        self._log_request(messages, self.model)
        
        try:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
            
            response = await client.post(
                self.api_url,
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            self._log_response(content, self.model)
            setattr(paragraph, self.result_field, content)
            
        except Exception as e:
            self._log(f"[LLM] Error calling {self.model}: {e}")
            setattr(paragraph, self.result_field, f"Error: {e}")
        
        return paragraph

    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self._close_client()


class ClaudeCompletion(LLMStageBase):
    """Text completion using Anthropic Claude API.
    
    Supports Claude models including:
    - Claude 3.5 Sonnet
    - Claude 3 Opus
    - Claude 3 Haiku
    """

    def __init__(
        self,
        api_url: str = "https://api.anthropic.com/v1/messages",
        api_key: str = "",
        model: str = "claude-3-sonnet-20240229",
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        result_field: str = "llm_completion",
    ) -> None:
        """Initialize Claude LLM stage.
        
        Args:
            api_url: API endpoint URL (default: Anthropic API)
            api_key: API key for authentication (default: empty)
            model: Model name to use (default: claude-3-sonnet-20240229)
            system_prompt: System prompt to set model behavior (default: empty)
            temperature: Sampling temperature (0.0-1.0, default: 0.7)
            max_tokens: Maximum tokens in response (default: 1000)
            result_field: Field name to store completion result (default: llm_completion)
        """
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.result_field = result_field
        super().__init__()

    async def resolve(self, paragraph: Paragraph) -> Paragraph:
        """Process paragraph through Claude API.
        
        Args:
            paragraph: Input paragraph to process.
            
        Returns:
            Paragraph with completion result added.
        """
        client = await self._ensure_client()
        
        # Build messages for Claude API
        messages = [{"role": "user", "content": paragraph.content}]
        
        # Claude API uses different structure
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        
        if self.system_prompt:
            payload["system"] = self.system_prompt
        
        self._log_request(messages, self.model)
        
        try:
            headers = {
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            }
            
            response = await client.post(
                self.api_url,
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            
            result = response.json()
            content = result["content"][0]["text"]
            
            self._log_response(content, self.model)
            setattr(paragraph, self.result_field, content)
            
        except Exception as e:
            self._log(f"[LLM] Error calling {self.model}: {e}")
            setattr(paragraph, self.result_field, f"Error: {e}")
        
        return paragraph

    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self._close_client()


class OllamaCompletion(LLMStageBase):
    """Text completion using Ollama API.
    
    Ollama is a local LLM server that supports various models.
    """

    def __init__(
        self,
        api_url: str = "http://localhost:11434/api/chat",
        model: str = "llama2",
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        result_field: str = "llm_completion",
        options: Dict[str, Any] = None,
    ) -> None:
        """Initialize Ollama LLM stage.
        
        Args:
            api_url: Ollama API endpoint URL (default: http://localhost:11434/api/chat)
            model: Model name to use (default: llama2)
            system_prompt: System prompt to set model behavior (default: empty)
            temperature: Sampling temperature (0.0-1.0, default: 0.7)
            max_tokens: Maximum tokens in response (default: 1000)
            result_field: Field name to store completion result (default: llm_completion)
            options: Additional Ollama options (default: None)
        """
        self.api_url = api_url
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.result_field = result_field
        self.options = options or {
            "num_predict": max_tokens,
            "temperature": temperature,
        }
        super().__init__()

    async def resolve(self, paragraph: Paragraph) -> Paragraph:
        """Process paragraph through Ollama API.
        
        Args:
            paragraph: Input paragraph to process.
            
        Returns:
            Paragraph with completion result added.
        """
        client = await self._ensure_client()
        
        # Build messages
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": paragraph.content})
        
        self._log_request(messages, self.model)
        
        try:
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": self.options,
            }
            
            response = await client.post(
                self.api_url,
                headers={"Content-Type": "application/json"},
                json=payload,
            )
            response.raise_for_status()
            
            result = response.json()
            content = result["message"]["content"]
            
            self._log_response(content, self.model)
            setattr(paragraph, self.result_field, content)
            
        except Exception as e:
            self._log(f"[LLM] Error calling Ollama {self.model}: {e}")
            setattr(paragraph, self.result_field, f"Error: {e}")
        
        return paragraph

    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self._close_client()


class LLMPlugin(Plugin):
    """Plugin for LLM API integration stages.
    
    Registers all LLM completion stages for use in pipelines.
    """

    def __init__(self, pmanager, **config) -> None:
        """Initialize LLM plugin.
        
        Args:
            pmanager: Plugin manager instance.
            **config: Plugin configuration.
        """
        super().__init__(pmanager, **config)
        self.register_pipelines(globals())


# Export all classes
__all__ = [
    "OpenAILikeCompletion",
    "ClaudeCompletion", 
    "OllamaCompletion",
    "LLMPlugin",
]
