THRESHOLDS = {
    "faithfulness": 0.85,
    "relevancy": 0.80,
    "hallucination_rate_max": 0.05
}

def evaluate_release_readiness(scores: dict) -> dict:
    failures = []
    for metric, threshold in THRESHOLDS.items():
        if metric == "hallucination_rate_max":
            actual = scores.get("hallucination_rate", 0)
            if actual > threshold:
                failures.append({"metric": "hallucination_rate",
                                  "actual": actual, "threshold": threshold})
        elif metric in scores:
            if scores[metric] < threshold:
                failures.append({"metric": metric,
                                  "actual": scores[metric], "threshold": threshold})
    return {"release_approved": len(failures) == 0,
            "failures": failures, "scores": scores}
