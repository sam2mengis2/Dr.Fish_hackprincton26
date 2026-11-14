"""Thin LLM wrapper.

provider == "mock": `is_mock` is True and agents use their deterministic fallbacks.
Otherwise it uses langchain's init_chat_model, so any provider langchain supports works:
    pip install langchain langchain-anthropic   # or langchain-openai
    export ANTHROPIC_API_KEY=...
"""
from __future__ import annotations

from typing import Optional, Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLM:
    def __init__(self, config: dict):
        c = config.get("llm", {})
        self.provider = c.get("provider", "mock")
        self.models = {
            "extractor": c.get("extractor_model"),
            "reasoner": c.get("reasoner_model"),
            "verifier": c.get("verifier_model"),
            "comms": c.get("comms_model"),
        }
        self.temperature = c.get("temperature")
        self._cache: dict = {}
        self.calls = 0

    @property
    def is_mock(self) -> bool:
        return self.provider == "mock"

    def _chat(self, role: str):
        if role not in self._cache:
            from langchain.chat_models import init_chat_model  # lazy: mock mode needs no langchain

            kwargs = {"model_provider": self.provider}
            if self.temperature is not None:  # newer Claude models reject non-default temperature
                kwargs["temperature"] = self.temperature
            self._cache[role] = init_chat_model(self.models[role], **kwargs)
        return self._cache[role]

    def structured(self, role: str, system: str, user: str, schema: Type[T]) -> T:
        """Return a validated pydantic object of type `schema`."""
        self.calls += 1
        try:  # newer models prefer native JSON-schema output over forced tool calls
            chat = self._chat(role).with_structured_output(schema, method="json_schema")
        except (TypeError, ValueError, NotImplementedError):
            chat = self._chat(role).with_structured_output(schema)
        return chat.invoke([("system", system), ("human", user)])

    def text(self, role: str, system: str, user: str) -> str:
        self.calls += 1
        out = self._chat(role).invoke([("system", system), ("human", user)])
        return out.content if isinstance(out.content, str) else str(out.content)


def make_llm(config: dict) -> LLM:
    return LLM(config)


def maybe(llm: Optional[LLM]) -> bool:
    """True when a real LLM should be used."""
    return llm is not None and not llm.is_mock
