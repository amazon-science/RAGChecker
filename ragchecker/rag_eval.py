import json
import numpy as np
from rank_eval import Qrels, Run, evaluate
from argparse import ArgumentParser
import os

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


def evaluate_retrieval(input_data):
    qrels = Qrels()
    run = Run()
    scores = []
    scores_gt = []
    qids = []
    gt_psgs = []
    pred_psgs = []


    for idx, item in enumerate(input_data):
        qid = str(item.get("query_id", idx))
        gt_ids = [str(psg.get("doc_id", psg.get("passage_id"))) for psg in item["gt_context"]]
        pred_ids = [str(psg.get("doc_id", psg.get("passage_id"))) for psg in item["retrieved_context"]]
        if not pred_ids:
            continue
        # scores are not available in bedrock kb, thus using fabricated scores
        # pred_scores = [psg["score"] for psg in item["retrieved_context"]]
        pred_scores = [1.-i/len(pred_ids) for i in range(len(pred_ids))]
        qids.append(qid)
        gt_psgs.append(gt_ids)
        pred_psgs.append(pred_ids)
        scores.append(pred_scores)
        scores_gt.append(
            [i+1 for i in range(len(gt_ids))]
        )
    qrels.add_multi(
        q_ids=qids,
        doc_ids=gt_psgs,
        scores=scores_gt
    )
    run.add_multi(
        q_ids=qids,
        doc_ids=pred_psgs,
        scores=scores,
    )
    results = evaluate(qrels, run, ["mrr", "recall@1", "recall@3", "recall"])
    results["recall@all"] = results["recall"]
    del results["recall"]

    return results


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


def evaluate_faithfulness(retrieved2response):
    faithfulness = []
    neutral_rate = []
    contradiction_rate = []
    for r2r in retrieved2response:
        if not r2r:  # no effective claims
            continue
        ret = 0.
        neutral = 0.
        contra = 0.
        for r2r_ci in r2r:
            if "Entailment" in r2r_ci:
                ret += 1
            elif all([v == 'Neutral' for v in r2r_ci]):
                neutral += 1
            else:
                contra += 1
        ret /= len(r2r)
        faithfulness.append(ret)
        neutral_rate.append(neutral / len(r2r))
        contradiction_rate.append(contra / len(r2r))
    
    return {"faithfulness": np.mean(faithfulness)}


def evaluate_source_attribution(citation_results):
    correctness = []
    for c in citation_results:
        if len(c) > 0:
            correctness.append(c.count('Entailment') / len(c))
    return {"citation_acc": np.mean(correctness)}


def evaluate_citations(data):
    acc_list = []
    for d, response_sents, response_claims, retrieved2response \
        in zip(data['input_data'], data['response_sents'], data['response_claims'], data['retrieved2response']):
            for citation in d['citations']:
                # attribute spans to sentences
                span_start, span_end = citation['generatedResponsePart']['textResponsePart']['span']['start'], \
                    citation['generatedResponsePart']['textResponsePart']['span']['end']
                sent_index_to_id = dict()
                for sent_id, sent_index in response_sents['sent_id_to_index'].items():
                    sent_index_to_id[sent_index] = sent_id
                
                sent_ids = set()
                for sent_index, sent in enumerate(response_sents['sents']):
                    if sent['is_blank']:
                        continue
                    if span_start <= sent['start'] <= sent['end'] <= span_end:
                        # the span covers the whole sentence
                        sent_ids.add(sent_index_to_id[sent_index])
                    elif sent['start'] <= span_start <= sent['end'] <= span_end:
                        # overlap, check whether the span covers more than half of the sent
                        if (sent['end'] - span_start) / (sent['end'] - sent['start']) > 0.5:
                            sent_ids.add(sent_index_to_id[sent_index])
                    elif span_start <= sent['start'] <= span_end <= sent['end']:
                        # overlap, check whether the span covers more than half of the sent
                        if (span_end - sent['start']) / (sent['end'] - sent['start']) > 0.5:
                            sent_ids.add(sent_index_to_id[sent_index])
                    elif sent['start'] <= span_start <= span_end <= sent['end']:
                        # the sent covers the whole span, not likely but possible
                        sent_ids.add(sent_index_to_id[sent_index])
                
                # further attribute spans to claims
                attributed_claim_indices = []
                for claim_index, claim in enumerate(response_claims):
                    if set(claim['attributed_sent_ids']).issubset(sent_ids):
                        attributed_claim_indices.append(claim_index)
                if len(attributed_claim_indices) == 0:
                    continue

                # get cited context
                cited_context_indices = []
                for ref in citation['retrievedReferences']:
                    for ret_cxt_index, ret_cxt in enumerate(d['retrieved_context']):
                        if ref['content']['text'] == ret_cxt['text']:
                            cited_context_indices.append(ret_cxt_index)
                
                # acc
                entail_cnt = 0
                for claim_index in attributed_claim_indices:
                    if any([retrieved2response[claim_index][ctx_i] == 'Entailment' for ctx_i in range(len(retrieved2response[claim_index]))]):
                        entail_cnt += 1
                acc_list.append(entail_cnt / len(attributed_claim_indices))
    return {"citation_acc": np.mean(acc_list)}


"""
* Correct Claims
    * with correct retrieval results
    * with Retrieved GT not cover enough context, helped by Retrieved Non-GT
    * with Hallucinate / self-knowledge

* Incorrect Claims
    * with Retrieved GT not cover enough context OR Misleading by Retrieved Noise
    * with Retrieved GT not cover enough context
    * with Misleading by Retrieved Noise
    * with Hallucinate / self-knowledge

* Missing Claims
    * with Generator not cover enough information OR Misleading by Retrieved Non-GT
    * with Generator not cover enough information
    * with Retrieved GT not cover enough context
    * GT Context labeling error or RefChecker error: missing claims should not contradict with retrieved GT
"""

def evaluate_diagnostic_metrics(data):
    def merge_ret(ret):
        """Merge results from multiple paragraphs"""
        if "Entailment" in ret:
            return "Entailment"
        if "Contradiction" in ret:
            return "Contradiction"
        return "Neutral"
    
    def permutation(ys):
        l2d = {
            "Entailment": 0,
            "Neutral": 1,
            "Contradiction": 2
        }
        return l2d[ys[0]] * 3 + l2d[ys[1]]

    C_label_mapping = {
        "C1": [0, 1, 2],
        "C2": [3, 6],
        "C3": [4, 5, 7, 8]
    }
    I_label_mapping = {
        "I1": [0],
        "I2": [1, 2],
        "I3": [3, 6],
        "I4": [4, 5, 7, 8]
    }
    M_label_mapping = {
        "M1": [2],
        "M2": [0, 1, 3],
        "M3": [4, 5],
        "M4": [6, 7, 8]
    }

    results = {
        "total_correct": 0,
        "%C1": 0,
        "%C2": 0,
        "%C3": 0,
        "total_incorrect": 0,
        "%I1": 0,
        "%I2": 0,
        "%I3": 0,
        "%I4": 0,
        "total_missing": 0,
        "%M1": 0,
        "%M2": 0,
        "%M3": 0,
        "%M4": 0,
    }
    assert len(data["input_data"]) == len(data["answer2response"])
    assert len(data["input_data"]) == len(data["response2answer"])
    assert len(data["input_data"]) == len(data["retrieved2response"])
    assert len(data["input_data"]) == len(data["retrieved2answer"])

    missing_label_cnt = {}
    for idx in range(len(data["input_data"])):
        correct_claims = []
        incorrect_claims = []
        missing_claims = [
            ci 
            for ci, res in enumerate(data["response2answer"][idx])
            if res != "Entailment"
        ]
        for ci, res in enumerate(data["answer2response"][idx]):
            if res == "Entailment":
                correct_claims.append(ci)
            else:
                incorrect_claims.append(ci)
        
        r_gt = []
        r_non_gt = []
        retrieved_context_cnt = len(data["input_data"][idx]["retrieved_context"])
        
        # label retrieved_gt based on whether it recall a gt_answer claim
        
        # answer_claim_cnt = len(data["retrieved2answer"][idx])
        # claim_recall_cnt = [0] * retrieved_context_cnt
        # for item in data["retrieved2answer"][idx]:
        #     assert len(item) == retrieved_context_cnt
        #     for ci, res in enumerate(item):
        #         if res == "Entailment":
        #             claim_recall_cnt[ci] += 1
        # for ci in range(retrieved_context_cnt):
        #     if claim_recall_cnt[ci] > answer_claim_cnt / 3:
        #         r_gt.append(ci)
        
        for item in data["retrieved2answer"][idx]:
            assert len(item) == retrieved_context_cnt
            for ci, res in enumerate(item):
                if res == "Entailment":
                    if ci not in r_gt:
                        r_gt.append(ci)

        for ci in range(retrieved_context_cnt):
            if ci not in r_gt:
                r_non_gt.append(ci)

        # if idx == 3:
        #     print(r_gt)
        #     print(r_non_gt)
        #     for i in range(retrieved_context_cnt):
        #         print(data["input_data"][idx]["retrieved_context"][i]["text"])
        #     for i in range(retrieved_context_cnt):
        #         ll = "gt"
        #         if i in r_non_gt:
        #             ll = "non_gt"
        #         print(ll)
        #     for ri in range(retrieved_context_cnt):
        #         ll = [data["retrieved2response"][idx][ci][ri] for ci in incorrect_claims]
        #         print(ll)
        #     for ri in range(retrieved_context_cnt):
        #         ll = [data["retrieved2answer"][idx][ci][ri] for ci in missing_claims]
        #         print(ll)


        
        #
        # label retrieved_gt based on gt_context
        #
        # gt_ctx_ids = [
        #     gt_ctx["doc_id"]
        #     for gt_ctx in data["input_data"][idx]["gt_context"]
        # ]
        # for ci, r_ctx in enumerate(data["input_data"][idx]["retrieved_context"]):
        #     if r_ctx["doc_id"] in gt_ctx_ids:
        #         r_gt.append(ci)
        #     else:
        #         r_non_gt.append(ci)
        
        for ci in correct_claims:
            y_r_gt = merge_ret([data["retrieved2response"][idx][ci][ri] for ri in r_gt])
            y_r_non_gt = merge_ret([data["retrieved2response"][idx][ci][ri] for ri in r_non_gt])
            label_id = permutation([y_r_gt, y_r_non_gt])
            for label in C_label_mapping:
                if label_id in C_label_mapping[label]:
                    results["%" + label] += 1
                    break
            if idx == 3:
                print("idx_3_correct_claim", label_id)
        results["total_correct"] += len(correct_claims)
        
        for ci in incorrect_claims:
            y_r_gt = merge_ret([data["retrieved2response"][idx][ci][ri] for ri in r_gt])
            y_r_non_gt = merge_ret([data["retrieved2response"][idx][ci][ri] for ri in r_non_gt])
            label_id = permutation([y_r_gt, y_r_non_gt])
            for label in I_label_mapping:
                if label_id in I_label_mapping[label]:
                    results["%" + label] += 1
                    break
            if idx == 3:
                print("idx_3_incorrect_claim", label_id)
        results["total_incorrect"] += len(incorrect_claims)

        for ci in missing_claims:
            y_r_gt = merge_ret([data["retrieved2answer"][idx][ci][ri] for ri in r_gt])
            y_r_non_gt = merge_ret([data["retrieved2answer"][idx][ci][ri] for ri in r_non_gt])
            label_id = permutation([y_r_gt, y_r_non_gt])
            for label in M_label_mapping:
                if label_id in M_label_mapping[label]:
                    results["%" + label] += 1
                    break
            if label_id in missing_label_cnt:
                missing_label_cnt[label_id] += 1
            else: 
                missing_label_cnt[label_id] = 1
            if idx == 3:
                print("idx_3_missing_claim", label_id)
        results["total_missing"] += len(missing_claims)

    print(missing_label_cnt)

    retriever_error_in_incorrect_labels = ["I2"]
    retriever_error_in_missing_labels = ["M3"]
    generator_error_in_incorrect_labels = ["I3", "I4"]
    generator_error_in_missing_labels = ["M2"]
    mixed_error_in_incorrect_labels = ["I1"]
    noisy_data_error_in_missing_labels = ["M1", "M4"]

    retriever_error_in_incorrect_cnt = 0
    retriever_error_in_missing_cnt = 0
    generator_error_in_incorrect_cnt = 0
    generator_error_in_missing_cnt = 0
    noisy_data_error_in_missing_cnt = 0

    for label in retriever_error_in_incorrect_labels:
        retriever_error_in_incorrect_cnt += results["%" + label]
    for label in retriever_error_in_missing_labels:
        retriever_error_in_missing_cnt += results["%" + label]
    
    for label in generator_error_in_incorrect_labels:
        generator_error_in_incorrect_cnt += results["%" + label]
    for label in generator_error_in_missing_labels:
        generator_error_in_missing_cnt += results["%" + label]
    
    for label in mixed_error_in_incorrect_labels:
        retriever_error_in_incorrect_cnt += results["%" + label] / 2
        generator_error_in_incorrect_cnt += results["%" + label] / 2
    for label in noisy_data_error_in_missing_labels:
        noisy_data_error_in_missing_cnt += results["%" + label]
    
    # total_cnt = results["total_correct"] + results["total_incorrect"] + results["total_missing"]
    # results["total_cnt"] = total_cnt
    total_answer_claims = sum([len(data["gt_answer_claims"][i]) for i in range(len(data["gt_answer_claims"]))])
    total_response_claims = sum([len(data["response_claims"][i]) for i in range(len(data["response_claims"]))])
    assert total_response_claims == results["total_correct"] + results["total_incorrect"]
    results["total_response_claims"] = total_response_claims
    results["retrieval_incorrect_rate"] = retriever_error_in_incorrect_cnt / total_response_claims
    results["generation_incorrect_rate"] = generator_error_in_incorrect_cnt / total_response_claims
    results["retrieval_missing_rate"] = retriever_error_in_missing_cnt / total_answer_claims
    results["generation_missing_rate"] = generator_error_in_missing_cnt / total_answer_claims
    results["noisy_data_missing_rate"] = noisy_data_error_in_missing_cnt / total_answer_claims

    # for label in C_label_mapping:
    #     results["%" + label] = results["%" + label] / results["total_correct"]
    # for label in I_label_mapping:
    #     results["%" + label] = results["%" + label] / results["total_incorrect"]
    # for label in M_label_mapping:
    #     results["%" + label] = results["%" + label] / results["total_missing"]

    return results

def evaluate_generator(data):
    context_utilization = []
    noise_sensitivity_in_relevant = []
    noise_sensitivity_in_irrelevant = []
    hallucination = []
    self_knowledge = []
    faithfulness = []
    response_claim_count = []
    response_acc = []

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
        
        # if len(correct_response_claims) + len(missing_answer_claims) > 0:
        #     context_utilization.append(len(correct_response_claims) / (len(correct_response_claims) + len(missing_answer_claims)))
        

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
            # Hallucination
            # hallucination.append(1 - tp / (len(incorrect_response_claims) + len(correct_response_claims)))
        
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
        
        # Response level accuracy
        if len(incorrect_response_claims) == 0:
            response_acc.append(1)
        else:
            response_acc.append(0)

    return {
        "context_utilization": np.mean(context_utilization),
        "noise_sensitivity_in_relevant": np.mean(noise_sensitivity_in_relevant),
        "noise_sensitivity_in_irrelevant": np.mean(noise_sensitivity_in_irrelevant),
        "hallucination": np.mean(hallucination),
        "self_knowledge": np.mean(self_knowledge),
        "faithfulness": np.mean(faithfulness),
        "claim_count": np.mean(response_claim_count),
        "response_acc": np.mean(response_acc)
    }

def eval(data):
    results = {
        "overall_metrics": {},
        "retriever_metrics": {},
        "generator_metrics": {}
    }
    # results.update(evaluate_retrieval(data["input_data"]))
    results["overall_metrics"].update(evaluate_correctness(data["answer2response"]))
    results["overall_metrics"].update(evaluate_completeness(data["response2answer"]))
    
    if results["overall_metrics"]['precision'] > 0 and results["overall_metrics"]['recall'] > 0:
        f1 = 2*results["overall_metrics"]['precision']*results["overall_metrics"]['recall'] / \
            (results["overall_metrics"]['precision']+results["overall_metrics"]['recall'])
    else:
        f1 = 0
    results["overall_metrics"]['f1'] = f1
    
    results["retriever_metrics"].update(evaluate_claim_recall(data["retrieved2answer"]))
    # results["generator_metrics"].update(evaluate_faithfulness(data["retrieved2response"]))
    # if args.check_citation:
        # results.update(evaluate_source_attribution(data["citation_results"]))
        # results["generator_metrics"].update(evaluate_citations(data))

    # results["diagnostic_metrics"].update(evaluate_diagnostic_metrics(data))
    results["generator_metrics"].update(evaluate_generator(data))

    return results

if __name__ == "__main__":    
    parser = ArgumentParser()
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--check_citation", action='store_true')
    args = parser.parse_args()
    
    if args.file.endswith('/'):
        args.file = args.file[:-1]
    
    all_results = {}
    if os.path.isdir(args.file):
        files = os.listdir(args.file)
        f1_scores = []
        for file in files:
            with open(os.path.join(args.file, file)) as fp:
                data = json.load(fp)
            results = eval(data)
            all_results.update({file[:-5]: results})
            for key in results:
                if isinstance(results[key], dict):
                    for k in results[key]:
                        if k == 'claim_count':
                            results[key][k] = int(results[key][k])
                        else:
                            results[key][k] = round(results[key][k] * 100, 1)
                else:
                    results[key] = round(results[key][k] * 100, 1)
            print(f'Results for {file[:-5]}:')
            print(json.dumps(results, indent=2))
            f1_scores.append(results['overall_metrics']['f1'])
        print(len(f1_scores), np.mean(f1_scores))
    else:
        with open(args.file) as fp:
            data = json.load(fp)
        results = eval(data)
        all_results.update({args.file: results})
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
        fout = open("result.txt", "w")
        fout.write(args.file + '\n')
        for metrics in results:
            for key in results[metrics]:
                fout.write("\t" + str(results[metrics][key]))
        fout.write("\n")
