import json
from argparse import ArgumentParser, RawTextHelpFormatter

DEFAULT_MODEL = "bedrock/meta.llama3-70b-instruct-v1:0"
DEFAULT_METRICS = "all_metrics"


def build_parser():
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
        '--extractor_name', type=str, default=DEFAULT_MODEL,
        help=f"Model used for extracting claims. Default: {DEFAULT_MODEL}"
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
        "--checker_name", type=str, default=DEFAULT_MODEL,
        help=f"Model used for checking whether the claims are factual. Default: {DEFAULT_MODEL}"
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
        '--metrics', type=str, nargs='+', default=[DEFAULT_METRICS],
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


    return parser


def get_args(args=None):
    return build_parser().parse_args(args)


def main():
    from .evaluator import RAGChecker
    from .container import RAGResults

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
