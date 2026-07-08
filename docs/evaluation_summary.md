# Evaluation Summary

## Metric Rubric

- **Faithfulness**: LLM-as-judge (deepseek-v4-pro) scoring. Checks whether every claim in the answer is supported by the retrieved context. Scale 0-1.
- **Correctness**: LLM-as-judge scoring. Checks how well the answer matches the ground truth. Scale 0-1.
- **Latency**: End-to-end time measured from question receipt to answer delivery.
- **Cost**: Estimated using DeepSeek API pricing (/usr/bin/bash.14/1M input, /usr/bin/bash.28/1M output).

## Target Thresholds

| Metric | Target | Actual | Status |
| --- | ---: | ---: | --- |
| Answer Faithfulness | >= 0.85 | 0.972 | PASS |
| Answer Correctness | >= 0.80 | 0.839 | PASS |
| P90 Latency | < 10s | ~7s | PASS |
| Out-of-scope Refusal Rate | — | 3/3 (100%) | PASS |

## Latest Evaluation Results

| Metric | Value |
| --- | ---: |
| Total evaluation questions | 21 |
| In-scope questions | 18 |
| Out-of-scope questions | 3 |
| Avg Faithfulness | 0.972 |
| Avg Correctness | 0.839 |
| Avg Retrieval Latency | ~150 ms |
| Avg Generation Latency | ~4200 ms |
| Avg Total Latency | ~4500 ms |
| Evaluation Method | LLM-as-judge (deepseek-v4-pro) |

## Sensitivity Analysis

Cost estimation across parameter combinations using DeepSeek API pricing:

| top_k | reranker | temp | Est. Latency | Cost/1K Calls |
| ---: | --- | ---: | ---: | ---: |
| 3 | off | 0.0 | 2.9s | /usr/bin/bash.18 |
| 5 | off | 0.0 | 3.5s | /usr/bin/bash.24 |
| 5 | on | 0.0 | 4.0s | /usr/bin/bash.24 |
| 10 | off | 0.0 | 5.0s | /usr/bin/bash.38 |
| 20 | on | 0.7 | 9.9s | /usr/bin/bash.68 |

Recommended configuration: top_k=5, reranker=off, temperature=0.0 (balanced cost/quality).

## Sample Redacted Log Entry



## Known Limitations

- The custom faithfulness metric (token-overlap) is a proxy. LLM-as-judge gives more accurate scores but adds latency and cost to evaluation.
- OCR for scanned PDFs is stubbed — PaddleOCR integration is planned but not active in the demo.
- Prompt-injection and PII controls are regex-based and should be complemented by enterprise DLP in production.
- The evaluation set (21 questions) is hand-crafted and should be expanded with human review for production use.
- Embedding uses bge-m3 which requires downloading a ~2.2GB model on first run.
