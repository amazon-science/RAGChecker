import json
import numpy as np
from argparse import ArgumentParser

def evaluate_claim_recall(retrieved2answer):
    claim_recall = []
    for r2a in retrieved2answer:
        if not r2a:  # no effective claims
            continue
        recall = 0.
        for r2a_ci in r2a:
            if "Entailment" in r2a_ci:
                recall += 1
        recall /= len(r2a)
        claim_recall.append(recall)
    
    psg_precision = []
    for r2a in retrieved2answer:
        if not r2a:  # no effective claims
            continue
        prec = 0.
        for psg_i in range(len(r2a[0])):
            if any([v == 'Entailment' for v in np.array(r2a)[:, psg_i]]):
                prec += 1
        psg_precision.append(prec / len(r2a[0]))
            
    return {"claim_recall": np.mean(claim_recall), "context_precision": np.mean(psg_precision)}


def evaluate_correctness(answer2response):
    correctness = []
    for a2r in answer2response:
        if not a2r:  # no effective claims
            continue
        correctness.append(a2r.count('Entailment') / len(a2r))
    return {"precision": np.mean(correctness)}


def evaluate_completeness(response2answer):
    completeness = []
    for r2a in response2answer:
        if not r2a:  # no effective claims
            continue
        completeness.append(r2a.count('Entailment') / len(r2a))
    return {"recall": np.mean(completeness)}


def evaluate_generator(data):
    context_utilization = []
    noise_sensitivity_in_relevant = []
    noise_sensitivity_in_irrelevant = []
    hallucination = []
    self_knowledge = []
    faithfulness = []
    response_claim_count = []

    for idx in range(len(data["input_data"])):
        correct_response_claims = []
        incorrect_response_claims = []
        for ci, res in enumerate(data["answer2response"][idx]):
            if res == "Entailment":
                correct_response_claims.append(ci)
            else:
                incorrect_response_claims.append(ci)

        correct_answer_claims = []
        missing_answer_claims = []
        for ci, res in enumerate(data["response2answer"][idx]):
            if res == "Entailment":
                correct_answer_claims.append(ci)
            else:
                missing_answer_claims.append(ci)
        
        relevant_chunks = []
        for ci in range(len(data["retrieved2answer"][idx][0])):
            if "Entailment" in np.array(data["retrieved2answer"][idx])[:, ci]:
                relevant_chunks.append(ci)
        
        # Context Utilization
        recall = 0.
        tp = 0.
        for ci, res in enumerate(data["retrieved2answer"][idx]):
            if "Entailment" in res:
                recall += 1
                if ci in correct_answer_claims:
                    tp += 1
        if recall > 0:
            context_utilization.append(tp / recall)

        # Noise Sensitivity
        if (len(incorrect_response_claims) + len(correct_response_claims)) > 0:
            tp = 0.
            in_relevant = 0.
            for ci, res in enumerate(data["retrieved2response"][idx]):
                if (ci in incorrect_response_claims) and ("Entailment" in res):
                    tp += 1
                    for c in relevant_chunks:
                        if res[c] == "Entailment":
                            in_relevant += 1
                            break
            noise_sensitivity_in_relevant.append(in_relevant / (len(incorrect_response_claims) + len(correct_response_claims)))
            noise_sensitivity_in_irrelevant.append((tp - in_relevant) / (len(incorrect_response_claims) + len(correct_response_claims)))
        
        # Hallucination and Self-knowledge
        if (len(incorrect_response_claims) + len(correct_response_claims)) > 0:
            tp = 0.
            for ci, res in enumerate(data["retrieved2response"][idx]):
                if (ci in correct_response_claims) and ("Entailment" not in res):
                    tp += 1
            self_knowledge.append(tp / (len(incorrect_response_claims) + len(correct_response_claims)))
            
            halu = 0.
            for ci, res in enumerate(data["retrieved2response"][idx]):
                if (ci in incorrect_response_claims) and ("Entailment" not in res):
                    halu += 1
            hallucination.append(halu / (len(incorrect_response_claims) + len(correct_response_claims)))
            
            faithfulness.append(1 - self_knowledge[-1] - hallucination[-1])
            response_claim_count.append(len(incorrect_response_claims) + len(correct_response_claims))

    return {
        "context_utilization": np.mean(context_utilization),
        "noise_sensitivity_in_relevant": np.mean(noise_sensitivity_in_relevant),
        "noise_sensitivity_in_irrelevant": np.mean(noise_sensitivity_in_irrelevant),
        "hallucination": np.mean(hallucination),
        "self_knowledge": np.mean(self_knowledge),
        "faithfulness": np.mean(faithfulness),
        "claim_count": np.mean(response_claim_count),
    }


if __name__ == "__main__":    
    parser = ArgumentParser()
    parser.add_argument("--file", type=str, required=True)
    args = parser.parse_args()
    
    with open(args.file) as fp:
        data = json.load(fp)

    results = {
        "overall_metrics": {},
        "retriever_metrics": {},
        "generator_metrics": {}
    }

    results["overall_metrics"].update(evaluate_correctness(data["answer2response"]))
    results["overall_metrics"].update(evaluate_completeness(data["response2answer"]))
    
    if results["overall_metrics"]['precision'] > 0 and results["overall_metrics"]['recall'] > 0:
        f1 = 2*results["overall_metrics"]['precision']*results["overall_metrics"]['recall'] / \
            (results["overall_metrics"]['precision']+results["overall_metrics"]['recall'])
    else:
        f1 = 0
    results["overall_metrics"]['f1'] = f1
    
    results["retriever_metrics"].update(evaluate_claim_recall(data["retrieved2answer"]))
    results["generator_metrics"].update(evaluate_generator(data))

    for key in results:
        if isinstance(results[key], dict):
            for k in results[key]:
                if k == 'claim_count':
                    results[key][k] = int(results[key][k])
                else:
                    results[key][k] = round(results[key][k] * 100, 1)
        else:
            results[key] = round(results[key][k] * 100, 1)
    print(f'Results for {args.file}:')
    print(json.dumps(results, indent=2))

