import os
import json
from argparse import ArgumentParser, RawTextHelpFormatter

import torch

from refchecker.extractor import LLMExtractor
from refchecker.checker import (
    LLMChecker, NLIChecker, AlignScoreChecker
)
from refchecker.aggregator import strict_agg, soft_agg, major_agg


def get_args():
    parser = ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument(
        "--input_path", type=str, required=True,
        help="Input path to the json file."
    )
    parser.add_argument(
        "--output_path", type=str, required=True,
        help="Output path to the result json file."
    )
    parser.add_argument(
        "--cache_dir", type=str, default="./.cache",
        help="Path to the cache directory. Default: ./.cache"
    )
    parser.add_argument(
        '--extractor_name', type=str, default="openai/meta-llama/Meta-Llama-3-70B-Instruct",
        help="Model used for extracting triplets. Default: claude3-sonnet."
    )
    parser.add_argument(
        '--extractor_max_new_tokens', type=int, default=1000,
        help="Max generated tokens of the extractor, set a larger value for longer documents. Default: 500"
    )
    parser.add_argument(
        "--checker_name", type=str, default="openai/meta-llama/Meta-Llama-3-70B-Instruct",
        help="Model used for checking whether the triplets are factual. "
        "Default: claude3-sonnet."
    )
    parser.add_argument(
        "--batch_size_extractor", type=int, default=32,
        help="Batch size for extractor."
    )
    parser.add_argument(
        "--batch_size_checker", type=int, default=32,
        help="Batch size for checker."
    )
    parser.add_argument(
        "--check_citations", action="store_true",
        help="Check citations for source attribution."
    )
    parser.add_argument(
        '--extractor_api_base', type=str
    )
    parser.add_argument(
        '--checker_api_base', type=str
    )
    # checking
    parser.add_argument(
        '--answer2response', action='store_true'
    )
    parser.add_argument(
        '--response2answer', action='store_true'
    )
    parser.add_argument(
        '--retrieved2answer', action='store_true'
    )
    parser.add_argument(
        '--retrieved2response', action='store_true'
    )
    parser.add_argument(
        '--openai_api_key', type=str
    )

    return parser.parse_args()


def main():
    args = get_args()
    
    if args.openai_key:
        os.environ['OPENAI_API_KEY'] = args.openai_key
    
    extract(args)
    check(args)


def extract(args):
    if os.path.exists(args.output_path) and 'response_claims' in json.load(open(args.output_path)):
        return
    
    # initialize extractor models
    print(args.openai_key, args.extractor_api_base)
    
    extractor = LLMExtractor(
        model=args.extractor_name, 
        batch_size=args.batch_size_extractor,
        api_base=args.extractor_api_base
    )
    
    # load data
    with open(args.input_path, "r") as fp:
        data = json.load(fp)
    input_data = data['input_data']

    responses = [item["response"] for item in input_data]
    questions = [item["query"] for item in input_data]

    
    # claim extraction
    print("Extracting claims...")
    response_extract_results = extractor.extract(
        batch_responses=responses,
        batch_questions=questions,
        max_new_tokens=args.extractor_max_new_tokens
    )
    response_claims = [[c.content for c in res.claims] for res in response_extract_results]
    
    # save results
    output_data = {
        "input_data": input_data,
        "gt_answer_claims": data['gt_answer_claims'],
        "response_claims": response_claims,
    }

    with open(args.output_path, "w") as fp:
        json.dump(output_data, fp, indent=2)
    torch.cuda.empty_cache()


def check(args):
    # load data
    with open(args.output_path, "r") as fp:
        data = json.load(fp)
    input_data = data['input_data']

    responses = [item["response"] for item in input_data]
    gt_answers = [item["gt_answer"] for item in input_data]
    questions = [item["query"] for item in input_data]
    retrieved = [
        [psg["text"] for psg in item["retrieved_context"]]
        for item in input_data
    ]
    
    # initialize checker models
    if args.checker_name == "nli":
        checker = NLIChecker(batch_size=args.batch_size_checker)
    elif args.checker_name == "alignscore":
        checker = AlignScoreChecker(batch_size=args.batch_size_checker)
    else:
        checker = LLMChecker(
            model=args.checker_name, 
            batch_size=args.batch_size_checker,
            api_base=args.checker_api_base
        )
    
    gt_answer_claims = data['gt_answer_claims']
    response_claims = data['response_claims']

    # checking
    if args.answer2response and 'answer2response' not in data:
        print("Checking GT answer -> response claims...")
        answer2response = checker.check(
            batch_claims=response_claims,
            batch_references=gt_answers, 
            batch_questions=questions,
            max_reference_segment_length=0
        )  # [num_items, num_claims]
        data['answer2response'] = answer2response
        json.dump(data, open(args.output_path, "w"), indent=2)
    
    if args.response2answer and 'response2answer' not in data:
        print("Checking response -> GT answer claims...")
        response2answer = checker.check(
            batch_claims=gt_answer_claims,
            batch_references=responses, 
            batch_questions=questions,
            max_reference_segment_length=0
        )  # [num_items, num_claims]
        data['response2answer'] = response2answer
        json.dump(data, open(args.output_path, "w"), indent=2)
    
    # we want fine-grained results on each passage
    if args.retrieved2answer and 'retrieved2answer' not in data:
        print("Checking retrieved -> GT answer claims...")
        retrieved2answer = checker.check(
            batch_claims=gt_answer_claims,
            batch_references=retrieved,
            batch_questions=questions,
            max_reference_segment_length=0,
            merge_psg=False
        )  # [num_items, num_claims, num_passages]
        data['retrieved2answer'] = retrieved2answer
        json.dump(data, open(args.output_path, "w"), indent=2)
    
    
    if args.retrieved2response and 'retrieved2response' not in data:
        print("Checking retrieved -> response claims...")
        retrieved2response = checker.check(
            batch_claims=response_claims,
            batch_references=retrieved,
            batch_questions=questions,
            max_reference_segment_length=0,
            merge_psg=False
        )  # [num_items, num_claims, num_passages]
        data["retrieved2response"] = retrieved2response
        json.dump(data, open(args.output_path, "w"), indent=2)


if __name__ == "__main__":
    main()
