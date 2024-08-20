import json
from typing import List
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from . import metrics

@dataclass_json
@dataclass
class RetrievedDoc:
    doc_id: str | None = None
    text: str = ""


@dataclass_json
@dataclass
class RAGResult:
    query_id: str
    query: str
    gt_answer: str
    response: str
    retrieved_context: List[RetrievedDoc] | None = None # Retrieved documents
    response_claims: List[List[str]] | None = None  # List of claims for the response
    gt_answer_claims: List[List[str]] | None = None  # List of claims for the ground truth answer
    answer2response: List[str] | None = None  # entailment results of answer -> response
    response2answer: List[str] | None = None  # entailment results of response -> answer
    retrieved2response: List[List[str]] | None = None  # entailment results of retrieved -> response
    retrieved2answer: List[List[str]] | None = None  # entailment results of retrieved -> answer
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass_json
@dataclass
class RAGResults:
    results: List[RAGResult] = field(default_factory=list)
    metrics: dict[str, dict[str, float]] = field(default_factory = lambda: {
        metrics.overall_metrics: {},
        metrics.retriever_metrics: {},
        metrics.generator_metrics: {}
    })

    def __repr__(self) -> str:
        metrics = '  ' + '\n  '.join(json.dumps(self.metrics, indent=2).split('\n'))
        return (
            f"RAGResults(\n  {len(self.results):,} RAG results,\n"
            f"  Metrics:\n{metrics}\n)"
        )

    def update(self, rag_result: List[RAGResult]):
        self.results.append(rag_result)
        self.metrics = {
            metrics.overall_metrics: {},
            metrics.retriever_metrics: {},
            metrics.generator_metrics: {}
        }
