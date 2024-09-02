import os
import json
from argparse import ArgumentParser, RawTextHelpFormatter

from .evaluator import RAGChecker
from .container import RAGResults
from .metrics import *


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
        '--extractor_name', type=str, default="bedrock/meta.llama3-70b-instruct-v1:0",
        help="Model used for extracting claims. Default: bedrock/meta.llama3-70b-instruct-v1:0"
    )
    parser.add_argument(
        '--extractor_api_base', type=str, default="bedrock/meta.llama3-70b-instruct-v1:0",
        help='API base URL for the extractor if using vllm deployed open source LLMs. Default: bedrock/meta.llama3-70b-instruct-v1:0'
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
        '--metrics', type=str, nargs='+', default=[all_metrics],
        help='Metrics to evaluate the results.'
    )
    parser.add_argument(
        '--openai_api_key', type=str
    )
    parser.add_argument(
        "--disable_joint_check", action="store_false", dest="joint_check",
        help="Disable joint checking of the claims."
    )
    parser.set_defaults(joint_check=True)
    parser.add_argument(
        "--joint_check_num", type=int, default=5
    )


    return parser.parse_args()


def main():
    args = get_args()
    evaluator = RAGChecker(
        extractor_name=args.extractor_name,
        checker_name=args.checker_name,
        extractor_max_new_tokens=args.extractor_max_new_tokens,
        extractor_api_base=args.extractor_api_base,
        checker_api_base=args.checker_api_base,
        batch_size_extractor=args.batch_size_extractor,
        batch_size_checker=args.batch_size_checker,
        openai_api_key=args.openai_api_key,
        joint_check=args.joint_check,
        joint_check_num=args.joint_check_num
    )
    with open(args.input_path, "r") as f:
        rag_results = RAGResults.from_json(f.read())
    evaluator.evaluate(rag_results, metrics=args.metrics, save_path=args.output_path)
    print(json.dumps(rag_results.metrics, indent=2))
    with open(args.output_path, "w") as f:
        f.write(rag_results.to_json(indent=2))


if __name__ == "__main__":
    main()
