import os
from typing import List

from refchecker.extractor import LLMExtractor
from refchecker.checker import (
    LLMChecker, NLIChecker, AlignScoreChecker
)
from loguru import logger
import numpy as np

from .container import RAGResults, RAGResult
from .metrics import *
from .computation import METRIC_FUNC_MAP

class RAGChecker():
    """
    RAGChecker class for evaluating RAG results.

    Parameters
    ----------
    extractor_name : str
        Model used for extracting claims. Default: "bedrock/meta.llama3-70b-instruct-v1:0".
    checker_name : str
        Model used for checking whether the claims are factual. Default: "bedrock/meta.llama3-70b-instruct-v1:0".
    extracto_max_new_tokens : int, optional
        Max generated tokens of the extractor, set a larger value for longer documents. Default: 1000.
    extractor_api_base : str, optional
        API base URL for the extractor if using vllm deployed open source LLMs.
    checker_api_base : str, optional
        API base URL for the checker if using vllm deployed open source LLMs.
    batch_size_extractor : int, optional
        Batch size for extractor. Default: 32.
    batch_size_checker : int, optional
        Batch size for checker. Default: 32.
    openai_api_key : str, optional
        OpenAI API key for using OpenAI models. Default: None.
    joint_check: bool, optional
        Enable joint checking of the claims. Default: True.
    joint_check_num: int, optional
        Number of claims to check jointly in one prompt. Default: 5.
    """
    def __init__(
        self,
        extractor_name="bedrock/meta.llama3-70b-instruct-v1:0",
        checker_name="bedrock/meta.llama3-70b-instruct-v1:0",
        extractor_max_new_tokens=1000,
        extractor_api_base=None,
        checker_api_base=None,
        batch_size_extractor=32,
        batch_size_checker=32,
        openai_api_key=None,
        joint_check=True,
        joint_check_num=5,
        sagemaker_client=None,
        **kwargs
    ):
        if openai_api_key:
            os.environ['OPENAI_API_KEY'] = openai_api_key
        self.extractor_max_new_tokens = extractor_max_new_tokens
        self.joint_check = joint_check
        self.joint_check_num = joint_check_num
        self.kwargs = kwargs
        self.sagemaker_client = sagemaker_client
        
        self.extractor = LLMExtractor(
            model=extractor_name, 
            batch_size=batch_size_extractor,
            api_base=extractor_api_base
        )
        if checker_name == "nli":
            self.checker = NLIChecker(batch_size=batch_size_checker)
        elif checker_name == "alignscore":
            self.checker = AlignScoreChecker(batch_size=batch_size_checker)
        else:
            self.checker = LLMChecker(
                model=checker_name, 
                batch_size=batch_size_checker,
                api_base=checker_api_base
            )
    
    def extract_claims(self, results: List[RAGResult], extract_type="gt_answer"):
        """
        Extract claims from the response and ground truth answer.

        Parameters
        ----------
        results : RAGResults
            RAGResults object.
        extract_type : str, optional
            Type of extraction, either 'gt_answer' or 'response'. Default: 'gt_answer'.
        """
        assert extract_type in ["gt_answer", "response"], \
            "extract_type should be either 'gt_answer' or 'response'."
        
        if extract_type == "gt_answer":
            results = [ret for ret in results if ret.gt_answer_claims is None]
            texts = [result.gt_answer for result in results]
        else:
            results = [ret for ret in results if ret.response_claims is None]
            texts = [result.response for result in results]
        if not results:
            return
        questions = [result.query for result in results]
        
        logger.info(f"Extracting claims for {extract_type} of {len(results)} RAG results.")
        extraction_results = self.extractor.extract(
            batch_responses=texts,
            batch_questions=questions,
            max_new_tokens=self.extractor_max_new_tokens,
            sagemaker_client=self.sagemaker_client,
            **self.kwargs
        )
        claims = [[c.content for c in res.claims] for res in extraction_results]
        for i, result in enumerate(results):
            if extract_type == "gt_answer":
                result.gt_answer_claims = claims[i]
            else:
                result.response_claims = claims[i]

    def check_claims(self, results: RAGResults, check_type="answer2response"):
        """
        Check the claims extracted from the response and ground truth answer.

        Parameters
        ----------
        results : RAGResults
            RAGResults object.
        check_type : str, optional
            Type of checking, either 'answer2response', 'response2answer', 'retrieved2answer',
            or 'retrieved2response'. Default: 'answer2response'.
        """
        match check_type:
            case "answer2response":
                results = [ret for ret in results.results if ret.answer2response is None]
                self.extract_claims(results, extract_type="response")
                claims = [ret.response_claims for ret in results]
                references = [ret.gt_answer for ret in results]
                merge_psg = True
            case "response2answer":
                results = [ret for ret in results.results if ret.response2answer is None]
                self.extract_claims(results, extract_type="gt_answer")
                claims = [ret.gt_answer_claims for ret in results]
                references = [ret.response for ret in results]
                merge_psg = True
            case "retrieved2answer":
                results = [ret for ret in results.results if ret.retrieved2answer is None]
                self.extract_claims(results, extract_type="gt_answer")
                claims = [ret.gt_answer_claims for ret in results]
                references = [[doc.text for doc in ret.retrieved_context] for ret in results]
                merge_psg = False
            case "retrieved2response":
                results = [ret for ret in results.results if ret.retrieved2response is None]
                self.extract_claims(results, extract_type="response")
                claims = [ret.response_claims for ret in results]
                references = [[doc.text for doc in ret.retrieved_context] for ret in results]
                merge_psg = False
            case _:
                raise ValueError(f"Invalid check_type: {check_type}")
        if not results:
            return

        logger.info(f"Checking {check_type} for {len(results)} RAG results.")
        checking_results = self.checker.check(
            batch_claims=claims,
            batch_references=references,
            batch_questions=[ret.query for ret in results],
            max_reference_segment_length=0,
            merge_psg=merge_psg,
            is_joint=self.joint_check,
            joint_check_num=self.joint_check_num,
            sagemaker_client=self.sagemaker_client,
            **self.kwargs
        )
        for i, result in enumerate(results):
            if check_type == "answer2response":
                result.answer2response = checking_results[i]
            elif check_type == "response2answer":
                result.response2answer = checking_results[i]
            elif check_type == "retrieved2answer":
                result.retrieved2answer = checking_results[i]
            else:
                result.retrieved2response = checking_results[i]
        
    def evaluate(self, results: RAGResults, metrics=all_metrics, save_path=None):
        """
        Evaluate the RAG results.

        Parameters
        ----------
        results : RAGResults
            RAGResults object.
        metrics : str | list[str], optional
            List of metrics to compute. Default: 'all'.
        save_path : str, optional
            Path to save the results. Default: None. Will perform progress checkpointing if provided.
        """ 
        # identify the metrics and required intermediate results
        if isinstance(metrics, str):
            metrics = [metrics]
        ret_metrics = set()
        requirements = set()
        for metric in metrics:
            if metric not in METRIC_REQUIREMENTS:
                if metric not in METRIC_GROUP_MAP:
                    raise ValueError(f"Invalid metric: {metric}.")
                ret_metrics.update(METRIC_GROUP_MAP[metric])
            else:
                ret_metrics.add(metric)
        for metric in ret_metrics:
            requirements.update(METRIC_REQUIREMENTS[metric])
        
        # compute the required intermediate results
        for requirement in requirements:
            self.check_claims(results, check_type=requirement)
            if save_path is not None:
                with open(save_path, "w") as f:
                    f.write(results.to_json(indent=2))

        # compute the metrics
        for metric in ret_metrics:
            for result in results.results:
                METRIC_FUNC_MAP[metric](result)
        
        # aggregate the metrics
        for group, group_metrics in METRIC_GROUP_MAP.items():
            if group == all_metrics:
                continue
            for metric in group_metrics:
                if metric in ret_metrics:
                    results.metrics[group][metric] = round(np.mean(
                        [result.metrics[metric] for result in results.results]
                    ) * 100, 1)

        return results.metrics