import os
import json
from argparse import ArgumentParser, RawTextHelpFormatter

from refchecker.extractor import LLMExtractor
from refchecker.checker import (
    LLMChecker, NLIChecker, AlignScoreChecker
)


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
        '--extractor_name', type=str,
        help="Model used for extracting claims."
    )
    parser.add_argument(
        '--extractor_api_base', type=str,
        help='API base URL for the extractor if using vllm deployed open source LLMs.'
    )
    parser.add_argument(
        '--extractor_max_new_tokens', type=int, default=1000,
        help="Max generated tokens of the extractor, set a larger value for longer documents. Default: 1000"
    )
    parser.add_argument(
        "--checker_name", type=str,
        help="Model used for checking whether the claims are factual. "
    )
    parser.add_argument(
        '--checker_api_base', type=str,
        help='API base URL for the checker if using vllm deployed open source LLMs.'
    )
    parser.add_argument(
        "--batch_size_extractor", type=int, default=32,
        help="Batch size for extractor."
    )
    parser.add_argument(
        "--batch_size_checker", type=int, default=32,
        help="Batch size for checker."
    )
    
    # checking options
    parser.add_argument(
        '--answer2response', action='store_true',
        help='Check claims in response using gt answer as reference.'
    )
    parser.add_argument(
        '--response2answer', action='store_true',
        help='Check claims in gt answer using response as reference.'
    )
    parser.add_argument(
        '--retrieved2answer', action='store_true',
        help='Check claims in gt answer using retrieved context as references.'
    )
    parser.add_argument(
        '--retrieved2response', action='store_true',
        help='Check claims in response using retrieved context as references.'
    )
    parser.add_argument(
        '--openai_api_key', type=str
    )
    parser.add_argument(
        '--joint_check', action='store_true'
    )
    parser.add_argument(
        '--with_rationale', action='store_true'
    )

    return parser.parse_args()


def main():
    args = get_args()
    
    if args.openai_api_key:
        os.environ['OPENAI_API_KEY'] = args.openai_api_key
    
    extract(args)
    check(args)


def extract(args):
    output_data = None
    if os.path.exists(args.output_path):
        output_data = json.load(open(args.output_path))
        if 'response_claims' in output_data and 'gt_answer_claims' in output_data:
            return
    
    # initialize extractor models    
    extractor = LLMExtractor(
        model=args.extractor_name, 
        batch_size=args.batch_size_extractor,
        api_base=args.extractor_api_base
    )
    
    # load data
    with open(args.input_path, "r") as fp:
        data = json.load(fp)
    input_data = data['input_data']
    
    if output_data is None:
        output_data = {"input_data": input_data}

    if 'response_claims' not in output_data:
        responses = [item["response"] for item in input_data]
        questions = [item["query"] for item in input_data]

        print("Extracting response claims...")
        response_extract_results = extractor.extract(
            batch_responses=responses,
            batch_questions=questions,
            max_new_tokens=args.extractor_max_new_tokens
        )
        response_claims = [[c.content for c in res.claims] for res in response_extract_results]
        output_data['response_claims'] = response_claims
        json.dump(output_data, open(args.output_path, "w"), indent=2)


    if 'gt_answer_claims' not in output_data:
        gt_answers = [item["gt_answer"] for item in input_data]
        questions = [item["query"] for item in input_data]
        print("Extracting ground truth answer claims...")
        response_extract_results = extractor.extract(
            batch_responses=gt_answers,
            batch_questions=questions,
            max_new_tokens=args.extractor_max_new_tokens
        )
        gt_answer_claims = [[c.content for c in res.claims] for res in response_extract_results]
        output_data['gt_answer_claims'] = gt_answer_claims
        json.dump(output_data, open(args.output_path, "w"), indent=2)


def check(args):
    # load data
    with open(args.output_path, "r") as fp:
        data = json.load(fp)
    input_data = data['input_data']

    responses = [item["response"] for item in input_data]
    gt_answers = [item["gt_answer"] for item in input_data]
    questions = [item["query"] for item in input_data]
    if args.retrieved2answer or args.retrieved2response:
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
            max_reference_segment_length=0,
            is_joint=args.joint_check,
            with_rationale=args.with_rationale
        )  # [num_items, num_claims]
        if answer2response is not None:
            data['answer2response'] = answer2response[0] if args.joint_check else answer2response
            json.dump(data, open(args.output_path, "w"), indent=2)
    
    if args.response2answer and 'response2answer' not in data:
        print("Checking response -> GT answer claims...")
        response2answer = checker.check(
            batch_claims=gt_answer_claims,
            batch_references=responses, 
            batch_questions=questions,
            max_reference_segment_length=0,
            is_joint=args.joint_check,
            with_rationale=args.with_rationale
        )  # [num_items, num_claims]
        if response2answer is not None:
            data['response2answer'] = response2answer[0] if args.joint_check else response2answer
            json.dump(data, open(args.output_path, "w"), indent=2)
    
    # we want fine-grained results on each passage
    if args.retrieved2answer and 'retrieved2answer' not in data:
        print("Checking retrieved -> GT answer claims...")
        retrieved2answer = checker.check(
            batch_claims=gt_answer_claims,
            batch_references=retrieved,
            batch_questions=questions,
            max_reference_segment_length=0,
            merge_psg=False,
            is_joint=args.joint_check,
            with_rationale=args.with_rationale
        )  # [num_items, num_claims, num_passages]
        if retrieved2answer is not None:
            data['retrieved2answer'] = retrieved2answer[0] if args.joint_check else retrieved2answer
            json.dump(data, open(args.output_path, "w"), indent=2)
    
    
    if args.retrieved2response and 'retrieved2response' not in data:
        print("Checking retrieved -> response claims...")
        retrieved2response = checker.check(
            batch_claims=response_claims,
            batch_references=retrieved,
            batch_questions=questions,
            max_reference_segment_length=0,
            merge_psg=False,
            is_joint=args.joint_check,
            with_rationale=args.with_rationale
        )  # [num_items, num_claims, num_passages]
        if retrieved2response is not None:
            data["retrieved2response"] = retrieved2response[0] if args.joint_check else retrieved2response
            json.dump(data, open(args.output_path, "w"), indent=2)


if __name__ == "__main__":
    main()
