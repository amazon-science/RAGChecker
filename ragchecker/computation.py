import numpy as np

from .container import RAGResult
from . import metrics


def to_bool(checking_results):
    if isinstance(checking_results, str):
        return checking_results == "Entailment"
    return np.array([to_bool(res) for res in checking_results])


def evaluate_precision(result: RAGResult):
    if metrics.precision in result.metrics:
        return
    assert result.answer2response is not None
    answer2response = to_bool(result.answer2response)
    if len(answer2response) > 0:
        result.metrics[metrics.precision] = np.mean(answer2response)
    else:
        result.metrics[metrics.precision] = 0.


def evaluate_recall(result: RAGResult):
    if "recall" in result.metrics:
        return
    assert result.response2answer is not None
    response2answer = to_bool(result.response2answer)
    if len(response2answer) > 0:
        result.metrics[metrics.recall] = np.mean(response2answer)
    else:
        result.metrics[metrics.recall] = 0.


def evaluate_f1(result: RAGResult):
    if "f1" in result.metrics:
        return
    evaluate_precision(result)
    evaluate_recall(result)
    precision = result.metrics[metrics.precision]
    recall = result.metrics[metrics.recall]
    if precision > 0 and recall > 0:
        result.metrics[metrics.f1] = 2 * precision * recall / (precision + recall)
    else:
        result.metrics[metrics.f1] = 0.


def evaluate_claim_recall(result: RAGResult):
    if metrics.claim_recall in result.metrics:
        return
    evaluate_retrieval(result)


def evaluate_context_precision(result: RAGResult):
    if metrics.context_precision in result.metrics:
        return
    evaluate_retrieval(result)


def evaluate_retrieval(result: RAGResult):
    """Evaluate retrieval metrics together as they share the same intermediate results."""
    assert result.retrieved2answer is not None
    retrieved2answer = to_bool(result.retrieved2answer)
    if len(retrieved2answer) > 0 and len(retrieved2answer[0]) > 0:
        claim_recalled = np.max(retrieved2answer, axis=1)
        result.metrics[metrics.claim_recall] = np.mean(claim_recalled)
        psg_useful = np.max(retrieved2answer, axis=0)
        result.metrics[metrics.context_precision] = np.mean(psg_useful)
    else:
        result.metrics[metrics.claim_recall] = 0.
        result.metrics[metrics.context_precision] = 0.


def evaluate_context_utilization(result: RAGResult):
    if "context_utilization" in result.metrics:
        return
    assert result.retrieved2answer is not None and result.response2answer is not None
    retrieved2answer = to_bool(result.retrieved2answer)
    response2answer = to_bool(result.response2answer)
    if len(retrieved2answer) > 0 and len(retrieved2answer[0]) > 0:
        claim_recalled = np.max(retrieved2answer, axis=1)
        if np.sum(claim_recalled) > 0:
            claim_used = claim_recalled & response2answer
            result.metrics[metrics.context_utilization] = np.sum(claim_used) / np.sum(claim_recalled)
        else:
            result.metrics[metrics.context_utilization] = 0.
    else:
        result.metrics[metrics.context_utilization] = 0.


def evaluate_noise_sensitivity_in_relevant(result: RAGResult):
    if metrics.noise_sensitivity_in_relevant in result.metrics:
        return
    evaluate_noise_sensitivity(result)


def evaluate_noise_sensitivity_in_irrelevant(result: RAGResult):
    if metrics.noise_sensitivity_in_irrelevant in result.metrics:
        return
    evaluate_noise_sensitivity(result)


def evaluate_noise_sensitivity(result: RAGResult):
    """Evaluate noise sensitivity metrics together as they share the same intermediate results."""
    assert result.retrieved2response is not None and result.answer2response is not None and \
        result.retrieved2answer is not None
    retrieved2response = to_bool(result.retrieved2response)
    answer2response = to_bool(result.answer2response)
    retrieved2answer = to_bool(result.retrieved2answer)
    if len(answer2response) > 0 and len(retrieved2response[0]) > 0 and len(retrieved2answer) > 0:
        relevant_retrieved = np.max(retrieved2answer, axis=0, keepdims=True)
        relevant_faithful = np.max(relevant_retrieved & retrieved2response, axis=1)
        irrelevant_retrieved = ~np.max(retrieved2answer, axis=0, keepdims=True)
        irrelevant_faithful = np.max(irrelevant_retrieved & retrieved2response, axis=1)
        irrelevant_faithful &= ~relevant_faithful  # to keep them exclusive

        incorrect = ~answer2response
        noise_sensitivity_in_relevant = np.mean(relevant_faithful & incorrect)
        noise_sensitivity_in_irrelevant = np.mean(irrelevant_faithful & incorrect)
        result.metrics[metrics.noise_sensitivity_in_relevant] = noise_sensitivity_in_relevant
        result.metrics[metrics.noise_sensitivity_in_irrelevant] = noise_sensitivity_in_irrelevant
    else:
        result.metrics[metrics.noise_sensitivity_in_relevant] = 0.
        result.metrics[metrics.noise_sensitivity_in_irrelevant] = 0.


def evaluate_hallucination(result: RAGResult):
    if "hallucination" in result.metrics:
        return
    evaluate_unfaithfulness(result)


def evaluate_self_knowledge(result: RAGResult):
    if "self_knowledge" in result.metrics:
        return
    evaluate_unfaithfulness(result)


def evaluate_unfaithfulness(result: RAGResult):
    """Evaluate hallucination and self-knowledge together as they share the same intermediate results."""
    assert result.retrieved2response is not None and result.answer2response is not None
    retrieved2response = to_bool(result.retrieved2response)
    answer2response = to_bool(result.answer2response)
    if  len(answer2response) > 0 and len(retrieved2response[0]) > 0:
        unfaithful = ~np.max(retrieved2response, axis=1)
        hallucination = np.mean(unfaithful & ~answer2response)
        self_knowledge = np.mean(unfaithful & answer2response)

        result.metrics[metrics.hallucination] = hallucination
        result.metrics[metrics.self_knowledge] = self_knowledge
    else:
        result.metrics[metrics.hallucination] = 0.
        result.metrics[metrics.self_knowledge] = 0.


def evaluate_faithfulness(result: RAGResult):
    assert result.retrieved2response is not None
    retrieved2response = to_bool(result.retrieved2response)
    if len(retrieved2response) > 0 and len(retrieved2response[0]) > 0:
        faithful = np.max(retrieved2response, axis=1)
        result.metrics[metrics.faithfulness] = np.mean(faithful)
    else:
        result.metrics[metrics.faithfulness] = 0.


METRIC_FUNC_MAP = {
    metrics.precision: evaluate_precision,
    metrics.recall: evaluate_recall,
    metrics.f1: evaluate_f1,
    metrics.claim_recall: evaluate_claim_recall,
    metrics.context_precision: evaluate_context_precision,
    metrics.context_utilization: evaluate_context_utilization,
    metrics.noise_sensitivity_in_relevant: evaluate_noise_sensitivity_in_relevant,
    metrics.noise_sensitivity_in_irrelevant: evaluate_noise_sensitivity_in_irrelevant,
    metrics.hallucination: evaluate_hallucination,
    metrics.self_knowledge: evaluate_self_knowledge,
    metrics.faithfulness: evaluate_faithfulness,
}
