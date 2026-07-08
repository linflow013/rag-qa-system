"""LLM-as-judge for accurate Faithfulness and Correctness evaluation."""

import json
import logging
import re
from typing import List

logger = logging.getLogger(__name__)

FAITHFULNESS_PROMPT = """Score the answer's FAITHFULNESS (0-1): whether ALL claims in the answer are supported by the retrieved context.

Context:
{context}

Answer:
{answer}

Score:
- 1.0: Every claim directly supported. No fabrications.
- 0.7-0.9: Most claims supported, minor elaboration.
- 0.4-0.6: Mixed, some claims supported, some not.
- 0.1-0.3: Mostly unsupported.
- 0.0: Totally fabricated.

Reply with ONLY: {{"score": X.XX, "reason": "one sentence"}}"""

CORRECTNESS_PROMPT = """Score the answer's CORRECTNESS (0-1): how well it matches the ground truth.

Ground Truth:
{ground_truth}

Answer:
{answer}

Score:
- 1.0: Matches ground truth completely, all key facts present.
- 0.7-0.9: Most key facts present, minor omissions.
- 0.4-0.6: Some facts present, significant omissions.
- 0.1-0.3: Mostly wrong or missing.
- 0.0: Completely wrong.

Reply with ONLY: {{"score": X.XX, "reason": "one sentence"}}"""


def _parse_json(text: str) -> dict:
    """Robust JSON parsing for LLM judge responses."""
    text = text.strip()
    # Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Extract JSON object
    m = re.search(r'\{[^{}]*"score"\s*:\s*[\d.]+[^{}]*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass

    # Extract just the score
    m = re.search(r'"score"\s*:\s*([\d.]+)', text)
    if m:
        return {"score": float(m.group(1)), "reason": "extracted"}

    # Fallback on empty
    if not text:
        logger.warning("Empty judge response")
    else:
        logger.warning("Could not parse judge response: %s", text[:100])

    return {"score": 0.5, "reason": "parse error"}


def llm_judge_faithfulness(llm_client, answer: str, context: List[str]) -> dict:
    context_text = "\n---\n".join(context[:5])
    prompt = FAITHFULNESS_PROMPT.format(
        context=context_text[:2000], answer=answer[:800]
    )
    result = llm_client.generate(
        prompt=prompt,
        system_prompt="You are an expert evaluator. Reply ONLY with valid JSON.",
        temperature=0.0,
        max_tokens=500,
    )
    return _parse_json(result.text)


def llm_judge_correctness(llm_client, answer: str, ground_truth: str) -> dict:
    prompt = CORRECTNESS_PROMPT.format(
        ground_truth=ground_truth[:800], answer=answer[:800]
    )
    result = llm_client.generate(
        prompt=prompt,
        system_prompt="You are an expert evaluator. Reply ONLY with valid JSON.",
        temperature=0.0,
        max_tokens=500,
    )
    return _parse_json(result.text)


def run_llm_evaluation(llm_client, test_cases: List[dict], retrieval_fn) -> dict:
    import time
    from src.generation.prompt_builder import build_prompt, build_system_prompt

    results = []
    for i, case in enumerate(test_cases):
        logger.info("Judge [%d/%d]: %s", i + 1, len(test_cases), case["question"][:50])

        r_start = time.time()
        retrieval_results = retrieval_fn(case["question"])
        r_latency = (time.time() - r_start) * 1000
        retrieved_texts = [r.text for r in retrieval_results]

        is_out_of_scope = case.get("relevant_doc") is None

        # Generate answer
        if is_out_of_scope and not retrieval_results:
            results.append({
                "index": i + 1, "question": case["question"],
                "is_out_of_scope": True,
                "faithfulness": 1.0, "correctness": 1.0,
                "retrieval_latency_ms": round(r_latency, 2),
                "generation_latency_ms": 0,
                "answer": "[REFUSED - out of scope]",
                "ground_truth": case["ground_truth"][:200],
            })
            continue

        lang = case.get("language", "en")
        gen_result = llm_client.generate(
            prompt=build_prompt(case["question"], retrieval_results),
            system_prompt=build_system_prompt(lang),
        )
        answer = gen_result.text

        # LLM-judge
        if is_out_of_scope:
            refusal_kw = ["cannot", "could not", "无法", "没有找到", "not found", "not covered"]
            properly_refused = any(kw in answer.lower() for kw in refusal_kw)
            faith_score = 1.0 if properly_refused else 0.0
            corr_score = 1.0 if properly_refused else 0.0
        else:
            faith_result = llm_judge_faithfulness(llm_client, answer, retrieved_texts) if retrieved_texts else {"score": 0.0}
            corr_result = llm_judge_correctness(llm_client, answer, case["ground_truth"])
            faith_score = faith_result.get("score", 0.5)
            corr_score = corr_result.get("score", 0.5)

        results.append({
            "index": i + 1, "question": case["question"],
            "language": case.get("language", "en"),
            "is_out_of_scope": is_out_of_scope,
            "faithfulness": round(faith_score, 4),
            "correctness": round(corr_score, 4),
            "retrieval_latency_ms": round(r_latency, 2),
            "generation_latency_ms": round(gen_result.latency_ms, 2),
            "answer": answer[:400],
            "ground_truth": case["ground_truth"][:200],
        })

    in_scope = [r for r in results if not r["is_out_of_scope"]]
    n = len(in_scope)
    avg_faith = sum(r["faithfulness"] for r in in_scope) / max(n, 1)
    avg_correct = sum(r["correctness"] for r in in_scope) / max(n, 1)
    oos = [r for r in results if r["is_out_of_scope"]]
    oos_pass = sum(1 for r in oos if r["faithfulness"] >= 0.5)

    return {
        "summary": {
            "total_test_cases": len(results),
            "in_scope_cases": n,
            "avg_faithfulness": round(avg_faith, 4),
            "faithfulness_target": 0.85,
            "faithfulness_pass": avg_faith >= 0.85,
            "avg_correctness": round(avg_correct, 4),
            "correctness_target": 0.80,
            "correctness_pass": avg_correct >= 0.80,
            "out_of_scope_cases": len(oos),
            "out_of_scope_correctly_refused": oos_pass,
            "evaluation_method": "LLM-as-judge (deepseek-v4-pro)",
        },
        "details": results,
    }
