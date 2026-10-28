"""12-factor configuration: every knob comes from the environment (or .env in dev)."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Dev-only default; Settings refuses to start with it when ENV=prod.
_DEV_SECRET = "dev-only-insecure-jwt-secret-change-me-0123456789"  # noqa: S105  # nosec B105
_DEV_PASSWORDS = {"admin123", "staff123"}
_BACKEND_DIR = Path(__file__).resolve().parents[2]
_REPO_ROOT = _BACKEND_DIR.parent


class Settings(BaseSettings):
    # Repo-root .env (shared with docker compose) first; backend/.env, if present, overrides it.
    model_config = SettingsConfigDict(
        env_file=(_REPO_ROOT / ".env", _BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Atlas"
    env: str = "dev"
    log_level: str = "INFO"
    data_dir: Path = Path("./data")
    samples_dir: Path = Path("../samples")
    static_dir: Path = Path("./static")  # Angular build output served by FastAPI
    seed_on_startup: bool = True

    # --- auth -------------------------------------------------------------
    jwt_secret: SecretStr = SecretStr(_DEV_SECRET)
    jwt_ttl_minutes: int = 8 * 60
    admin_username: str = "admin"
    admin_password: SecretStr = SecretStr("admin123")
    staff_username: str = "staff"
    staff_password: SecretStr = SecretStr("staff123")

    # --- LLM providers ----------------------------------------------------
    anthropic_api_key: SecretStr | None = None
    # Only Haiku is exposed: the registry rejects any model not in this allow-list, so the
    # restriction holds server-side, not just in the UI dropdown.
    claude_model: str = "claude-haiku-5-5"
    claude_model_options: list[str] = ["claude-haiku-5-5"]
    claude_fast_model: str = "claude-haiku-5-5"  # query rewrite, planner, judge
    claude_effort: str = "low"  # chat-style RAG does not need deep reasoning
    claude_refusal_fallback: bool = True

    openweight_base_url: str = "https://router.huggingface.co/v1"
    openweight_api_key: SecretStr | None = Field(default=None, alias="HF_TOKEN")
    openweight_model: str = "Qwen/Qwen2.5-72B-Instruct"
    openweight_model_options: list[str] = [
        "Qwen/Qwen2.5-72B-Instruct",
        "meta-llama/Llama-3.3-70B-Instruct",
    ]

    llm_max_tokens: int = 2048
    llm_timeout_s: float = 60.0

    # --- retrieval ---------------------------------------------------------
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    # MiniLM: same top-1 as bge-reranker-base on our probe set at ~6x lower CPU latency
    # (0.6s vs 3.6s per query). Swap via RERANKER_MODEL - see docs/adr/0002.
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_enabled: bool = True
    dense_top_k: int = 20
    sparse_top_k: int = 20
    rrf_k: int = 60
    rerank_pool: int = 30
    context_top_k: int = 6
    min_rerank_score: float | None = None  # global override of per-agent thresholds

    # --- memory / context --------------------------------------------------
    history_turns_verbatim: int = 6
    context_token_budget: int = 6000

    # --- guardrails / evals ------------------------------------------------
    max_input_chars: int = 4000
    rate_limit_per_minute: int = 30
    online_eval_enabled: bool = True

    # --- observability -----------------------------------------------------
    otel_exporter: str = "none"  # none | console | otlp

    @model_validator(mode="after")
    def _refuse_insecure_prod(self) -> "Settings":
        """Fail fast: never boot a deployed instance with dev credentials."""
        if self.claude_model not in self.claude_model_options:
            raise ValueError(
                f"CLAUDE_MODEL={self.claude_model} is not in CLAUDE_MODEL_OPTIONS "
                f"{self.claude_model_options}"
            )
        if self.env == "prod":
            secret = self.jwt_secret.get_secret_value()
            if secret == _DEV_SECRET or len(secret) < 32:
                raise ValueError("JWT_SECRET must be set to a random value of >= 32 chars")
            passwords = {
                self.admin_password.get_secret_value(),
                self.staff_password.get_secret_value(),
            }
            if passwords & _DEV_PASSWORDS:
                raise ValueError("ADMIN_PASSWORD / STAFF_PASSWORD must be changed in prod")
        return self

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def sqlite_path(self) -> Path:
        return self.data_dir / "atlas.db"

    @property
    def checkpoint_path(self) -> Path:
        return self.data_dir / "checkpoints.db"


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return settings
