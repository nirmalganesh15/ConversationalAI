"""Builds the configured providers once; the UI picks one per request by key."""

from pydantic import BaseModel

from app.core.config import Settings
from app.core.llm.anthropic_provider import AnthropicProvider
from app.core.llm.base import LLMProvider
from app.core.llm.fake import FakeProvider
from app.core.llm.openweight_provider import OpenWeightProvider


class ProviderInfo(BaseModel):
    key: str
    label: str
    models: list[str]
    default_model: str
    available: bool


class ProviderRegistry:
    def __init__(self, providers: dict[str, LLMProvider], infos: list[ProviderInfo]) -> None:
        self._providers = providers
        self.infos = infos

    @classmethod
    def from_settings(cls, s: Settings) -> "ProviderRegistry":
        providers: dict[str, LLMProvider] = {}
        if s.anthropic_api_key:
            providers["claude"] = AnthropicProvider(
                api_key=s.anthropic_api_key.get_secret_value(),
                main_model=s.claude_model,
                fast_model=s.claude_fast_model,
                effort=s.claude_effort,
                refusal_fallback=s.claude_refusal_fallback,
                default_max_tokens=s.llm_max_tokens,
                timeout_s=s.llm_timeout_s,
            )
        if s.openweight_api_key:
            providers["open"] = OpenWeightProvider(
                base_url=s.openweight_base_url,
                api_key=s.openweight_api_key.get_secret_value(),
                model=s.openweight_model,
                default_max_tokens=s.llm_max_tokens,
                timeout_s=s.llm_timeout_s,
            )
        if not providers:
            # No keys configured: keep the app demoable end-to-end with a canned model.
            providers["fake"] = FakeProvider()
        infos = [
            ProviderInfo(
                key="claude",
                label="Claude (Anthropic)",
                models=s.claude_model_options,
                default_model=s.claude_model,
                available="claude" in providers,
            ),
            ProviderInfo(
                key="open",
                label="Open-weight (HF Inference)",
                models=s.openweight_model_options,
                default_model=s.openweight_model,
                available="open" in providers,
            ),
        ]
        if "fake" in providers:
            infos.append(
                ProviderInfo(
                    key="fake",
                    label="Offline demo",
                    models=["fake"],
                    default_model="fake",
                    available=True,
                )
            )
        return cls(providers, infos)

    def get(self, key: str, model: str | None = None) -> LLMProvider:
        provider = self._providers.get(key) or next(iter(self._providers.values()))
        if model and model != provider.model_for("main"):
            info = next((i for i in self.infos if i.key == key), None)
            if info and model in info.models:  # only allow-listed models
                return provider.with_main_model(model)  # type: ignore[attr-defined]
        return provider

    def judge(self) -> LLMProvider:
        """Cheapest available provider for LLM-as-judge and classifiers."""
        return self._providers.get("claude") or next(iter(self._providers.values()))
