"""USD per 1M tokens. Used for the per-answer cost estimate shown in the UI.

Prices are list prices at time of writing; the table is data, not logic, so updating
it never touches provider code. Unknown models cost 0 rather than guessing.
"""

from app.core.llm.base import Usage

# model -> (input, output, cache_read, cache_write)
_PRICES: dict[str, tuple[float, float, float, float]] = {
    "claude-opus-5-5": (4.00, 20.00, 0.20, 5.00),
    "claude-sonnet-5-5": (2.00, 10.00, 0.20, 2.50),
    "claude-haiku-5-5": (0.10, 0.50, 0.01, 0.125),
    # HF Inference Providers - varies by routed provider; representative figures.
    "Qwen/Qwen2.5-72B-Instruct": (0.40, 0.40, 0.0, 0.0),
    "meta-llama/Llama-3.3-70B-Instruct": (0.40, 0.40, 0.0, 0.0),
}


def estimate_cost(model: str, usage: Usage) -> float:
    price_in, price_out, price_cache_read, price_cache_write = _PRICES.get(model, (0, 0, 0, 0))
    total = (
        usage.input_tokens * price_in
        + usage.output_tokens * price_out
        + usage.cache_read_tokens * price_cache_read
        + usage.cache_write_tokens * price_cache_write
    )
    return round(total / 1_000_000, 6)
