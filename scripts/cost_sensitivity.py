#!/usr/bin/env python
"""
Cost sensitivity analysis for RAG system parameters.

Varies top_k, reranker, and temperature to measure cost-quality trade-offs.

Usage:
    python scripts/cost_sensitivity.py
"""
import json
import sys

sys.path.insert(0, ".")

from src.logging.cost_tracker import run_sensitivity_analysis


def main():
    print("=" * 70)
    print("  Cost Sensitivity Analysis")
    print("  Parameters: top_k, reranker_enabled, temperature")
    print("=" * 70)
    print()

    results = run_sensitivity_analysis(
        top_k_values=[3, 5, 10, 20],
        reranker_options=[False, True],
        temperature_values=[0.0, 0.3, 0.7],
    )

    # Print table
    header = (
        f"{'top_k':>6} | {'reranker':>8} | {'temp':>5} | "
        f"{'input_tok':>9} | {'output_tok':>10} | "
        f"{'latency(s)':>10} | {'cost/1K(USD)':>13}"
    )
    print(header)
    print("-" * len(header))

    for row in results:
        print(
            f"{row['top_k']:>6} | {str(row['reranker_enabled']):>8} | "
            f"{row['temperature']:>5.1f} | {row['est_input_tokens']:>9} | "
            f"{row['est_output_tokens']:>10} | {row['est_latency_seconds']:>10.1f} | "
            f"${row['cost_per_1000_calls_usd']:>11.2f}"
        )

    print()
    print("Assumptions:")
    print("  - ~200 tokens per chunk in context + 300 tokens prompt overhead")
    print("  - Output tokens scale with temperature (longer answers at higher temp)")
    print("  - Reranker adds ~500ms latency but does not affect token count")
    print("  - DeepSeek pricing: $0.14/1M input, $0.28/1M output tokens")
    print()

    # Save results
    output_path = "logs/sensitivity_analysis.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Results saved to: {output_path}")

    # Recommendation
    print()
    print("Recommended configuration (balanced cost/quality):")
    best = min(results, key=lambda r: r["cost_per_1000_calls_usd"])
    mid_latency = [r for r in results if 4.0 <= r["est_latency_seconds"] <= 7.0]
    rec = mid_latency[0] if mid_latency else best
    print(f"  top_k={rec['top_k']}, reranker={'on' if rec['reranker_enabled'] else 'off'}, "
          f"temperature={rec['temperature']}")
    print(f"  Estimated: {rec['est_latency_seconds']}s latency, "
          f"${rec['cost_per_1000_calls_usd']}/1K calls")


if __name__ == "__main__":
    main()
