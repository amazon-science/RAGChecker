import json
import os
import numpy as np
from scipy import stats
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go

rag_systems = [
    'bm25_gpt_4',
    'bm25_llama3_8b',
    'bm25_llama3_70b',
    'bm25_mixtral_8x7b',
    'e5_mistral_gpt_4',
    'e5_mistral_llama3_8b',
    'e5_mistral_llama3_70b',
    'e5_mistral_mixtral_8x7b',
]

evaluation_metrics = {
    "trulens": [
        'groundedness',
        'answer_relevance'
    ],
    "ares": [
        "Answer Relevance Scores",
        "Answer Faithfulness Scores"
    ],
    "ragas": [
        'faithfulness', 
        'answer_correctness', 
        'answer_similarity', 
        'answer_relevancy'
    ],
    "crud": [
        'bleu-avg',
        'rouge-L',
        'bertScore',
        'QA_avg_F1',
        'QA_recall'
    ],
    "ragchecker": [
        'correctness_label',
        'completeness_label',
        'overall_label'
    ],
    "human": [
        'correctness_label',
        'completeness_label',
        'overall_label'
    ],
}

def correlation(a, b):
    pearson = round(stats.pearsonr(a, b)[0] * 100, 2)
    spearman = round(stats.spearmanr(a, b)[0] * 100, 2)
    return pearson, spearman

def eval(baseline, use_llama3=False):
    if baseline == "human":
        baseline_data = json.load(open(f"human_labeled_data.json"))
    else:
        if use_llama3 and baseline != "ragchecker":
            baseline_data = json.load(open(f"baseline_{baseline}_llama3.json"))
        else:
            baseline_data = json.load(open(f"baseline_{baseline}.json"))
    
    sample_result = {
        metric: {}
        for metric in evaluation_metrics[baseline]
    }
    
    # baseline
    nan_cnt = 0
    for metric in evaluation_metrics[baseline]:
        if baseline == "human":
            delta = np.array([data[metric] for data in baseline_data])
            # there are 280 meta evaluation data instance, each instance is assigned to 2 annotators
            # so there are a total of 560 human labels
            assert(len(delta) == 560)
            
            delta_norm = delta
        else:
            # linear normalize the prediction score to [-2, 2] to fit human label range
            delta = np.array([data["model2"][baseline][metric] - data["model1"][baseline][metric] for data in baseline_data])
            assert(len(delta) == 280)
            
            mini = np.nanmin(delta)
            maxi = np.nanmax(delta)
            median = np.nanmedian(delta)
            
            # fix nan with median
            for idx in range(len(delta)):
                if np.isnan(delta[idx]):
                    nan_cnt += 1
                    delta[idx] = median
                    
            if maxi - mini != 0:
                delta_norm = ((delta - mini) / (maxi - mini) - 0.5) * 4
            else:
                delta_norm = delta

        for idx, data in enumerate(baseline_data):
            sample_result[metric].update({idx: delta_norm[idx]})

    print(f"nan cnt in {baseline}:", nan_cnt)
    
    return {baseline: sample_result}

results = {}
for baseline in evaluation_metrics:
    results.update(eval(baseline, use_llama3=True))

eval_results = {}
for human_metric in evaluation_metrics["human"]:
    eval_result = {}
    human_data = results["human"][human_metric]

    plot_data = {}
    human_label = np.array(list(human_data.values()))
    plot_data["human_label"] = human_label
    for baseline in results:
        eval_result[baseline] = {}
        if baseline == "human":
            # human sanity check of correlation
            anno1 = np.array([human_data[idx] for idx in human_data if idx % 2 == 0])
            anno2 = np.array([human_data[idx] for idx in human_data if idx % 2 == 1])
            eval_result[baseline]["sanity check"] = correlation(anno1, anno2)
            continue
        for metric in results[baseline]:    
            if baseline == "ragchecker" and metric != human_metric:
                continue
            else:
                # copy 280 baseline score instances to calculate correlation with 560 human labels
                baseline_data = {idx: results[baseline][metric][idx // 2] for idx in human_data}
            x = np.array(list(baseline_data.values()))
            eval_result[baseline][metric] = correlation(x, human_label)
            if baseline == "ragchecker" or baseline == "ragas" and metric == "answer_similarity":
                plot_data[f"{baseline}"] = x
    eval_results[f"correlation with {human_metric}"] = eval_result

    # violin plot to compare RAGChecker with RAGAS Answer Similarity
    data = {
        f"human_{human_metric[:-6]}": np.tile(plot_data["human_label"], 2),
        "prediction": np.concatenate((plot_data["ragchecker"], plot_data["ragas"])),
        "baseline": np.array([f"ragchecker_{human_metric[:-6]}"] * human_label.shape[0] + ["ragas_answer_similarity"] * human_label.shape[0])
    }
    df = pd.DataFrame(data=data, index=list(human_data.keys()) * 2)

    fig = go.Figure()    
    fig.add_trace(go.Violin(x = df[f"human_{human_metric[:-6]}"][ df["baseline"] == "ragas_answer_similarity" ],
                            y = df["prediction"][ df["baseline"] == "ragas_answer_similarity" ],
                            legendgroup=f"ragas", scalegroup=f"ragas", name="RAGAS Answer Similarity",
                            side="negative",
                            line_color="blue")
                 )
    fig.add_trace(go.Violin(x = df[f"human_{human_metric[:-6]}"][ df["baseline"] == f"ragchecker_{human_metric[:-6]}" ],
                            y = df["prediction"][ df["baseline"] == f"ragchecker_{human_metric[:-6]}" ],
                            legendgroup=f"ragchecker", scalegroup=f"ragchecker", name=f"RAGChecker {human_metric[:-6]}",
                            side="positive",
                            line_color="red")
                 )
    fig.update_traces(meanline_visible=True)
    fig.update_layout(violingap=0, violingroupgap=0, violinmode='overlay')
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title=f"Human {human_metric[:-6]}",
        yaxis_title="Prediction Score",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    fig.write_image(f"human_{human_metric[:-6]}.png", scale=5)

json.dump(eval_results, open("meta_eval_results.json", "w"), indent=2)