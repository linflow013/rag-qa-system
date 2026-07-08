"""Token cost estimation and sensitivity analysis."""

import json
import os
from typing import Dict, List

# Pricing per 1M tokens (approximate, in USD)
# Reference: DeepSeek API pricing
PRICING = {
    "deepseek-chat": {"input": 0.14, "output": 0.28},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}


class CostTracker:
    """Tracks per-request and cumulative token costs."""

    def __init__(self, model_name: str = "deepseek-chat"):
        self.model_name = model_name
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_requests = 0
        self.pricing = PRICING.get(model_name, {"input": 0.0, "output": 0.0})

    def record(self, input_tokens: int, output_tokens: int):
        """Record token usage for one request."""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_requests += 1

    def per_request_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for a single request."""
        cost = (
            input_tokens / 1_000_000 * self.pricing["input"]
            + output_tokens / 1_000_000 * self.pricing["output"]
        )
        return cost

    def cost_per_1000_calls(
        self, avg_input: int, avg_output: int
    ) -> float:
        """Estimate cost for 1000 calls."""
        return self.per_request_cost(avg_input, avg_output) * 1000

    def cumulative_cost(self) -> float:
        """Total cost so far."""
        return self.per_request_cost(
            self.total_input_tokens, self.total_output_tokens
        )

    def stats(self) -> dict:
        return {
            "model": self.model_name,
            "total_requests": self.total_requests,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "avg_input_tokens": (
                self.total_input_tokens / self.total_requests
                if self.total_requests > 0 else 0
            ),
            "avg_output_tokens": (
                self.total_output_tokens / self.total_requests
                if self.total_requests > 0 else 0
            ),
            "cumulative_cost_usd": round(self.cumulative_cost(), 6),
            "pricing_per_1M": self.pricing,
        }


def run_sensitivity_analysis(
    top_k_values: List[int] = None,
    reranker_options: List[bool] = None,
    temperature_values: List[float] = None,
) -> List[dict]:
    """
    Generate a cost/quality sensitivity analysis table.
    Uses estimated token counts based on top_k and temperature.
    """
    if top_k_values is None:
        top_k_values = [3, 5, 10, 20]
    if reranker_options is None:
        reranker_options = [False, True]
    if temperature_values is None:
        temperature_values = [0.0, 0.3, 0.7]

    results = []

    for top_k in top_k_values:
        for reranker in reranker_options:
            for temp in temperature_values:
                # Estimate tokens based on top_k
                # ~200 tokens per chunk in context + 150 for prompt + 200 for output
                est_input = 300 + top_k * 200
                est_output = 200 if temp == 0.0 else 200 + int(temp * 100)

                # Reranker adds ~300ms latency but doesn't change token count
                est_latency = 2.0 + top_k * 0.3 + (0.5 if reranker else 0) + temp * 2.0

                tracker = CostTracker("deepseek-chat")
                cost = tracker.cost_per_1000_calls(est_input, est_output)

                results.append({
                    "top_k": top_k,
                    "reranker_enabled": reranker,
                    "temperature": temp,
                    "est_input_tokens": est_input,
                    "est_output_tokens": est_output,
                    "est_latency_seconds": round(est_latency, 1),
                    "cost_per_1000_calls_usd": round(cost, 2),
                })

    return results
