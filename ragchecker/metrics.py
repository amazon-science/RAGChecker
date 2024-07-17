overall_metrics = "overall_metrics"
precision = "precision"
recall = "recall"
f1 = "f1"

retriever_metrics = "retriever_metrics"
claim_recall = "claim_recall"
context_precision = "context_precision"
context_utilization = "context_utilization"

generator_metrics = "generator_metrics"
noise_sensitivity_in_relevant = "noise_sensitivity_in_relevant"
noise_sensitivity_in_irrelevant = "noise_sensitivity_in_irrelevant"
hallucination = "hallucination"
self_knowledge = "self_knowledge"
faithfulness = "faithfulness"

all_metrics = "all_metrics"


METRIC_GROUP_MAP = {
    overall_metrics: [precision, recall, f1],
    retriever_metrics: [claim_recall, context_precision],
    generator_metrics: [
        context_utilization, noise_sensitivity_in_relevant, noise_sensitivity_in_irrelevant,
        hallucination, self_knowledge, faithfulness
    ],
    all_metrics: [
        precision, recall, f1, claim_recall, context_precision,
        context_utilization, noise_sensitivity_in_relevant, noise_sensitivity_in_irrelevant,
        hallucination, self_knowledge, faithfulness
    ]
}


METRIC_REQUIREMENTS = {
    # overall metrics
    precision: ["answer2response"],
    recall: ["response2answer"],
    f1: ["answer2response", "response2answer"],
    # retriever metrics
    claim_recall: ["retrieved2answer"],
    context_precision: ["retrieved2answer"],
    # generator metrics
    context_utilization: ["retrieved2answer", "response2answer"],
    noise_sensitivity_in_relevant: ["retrieved2response", "answer2response", "retrieved2answer"],
    noise_sensitivity_in_irrelevant: ["retrieved2response", "answer2response", "retrieved2answer"],
    hallucination: ["retrieved2response", "answer2response"],
    self_knowledge: ["retrieved2response", "answer2response"],
    faithfulness: ["retrieved2response"],
}