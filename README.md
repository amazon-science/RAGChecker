# RAGChecker: A Fine-grained Framework For Diagnosing RAG

## 🚀 Quick Start

### Setup Environment

```bash
pip install refchecker
python -m spacy download en_core_web_sm
pip install -r requirements
```


## Checking

If you are using AWS Bedrock version of Llama3 for the claim extractor and checker, use the following command:


```bash
python checking.py \
    --input_path=<path_to_generated_responses> \
    --output_path=<path_to_checking_results> \
    --extractor_name=bedrock/meta.llama3-70b-instruct-v1:0 \
    --checker_name=bedrock/meta.llama3-70b-instruct-v1:0 \
    --batch_size_extractor=64 \
    --batch_size_checker=128 \
    --answer2response \
    --response2answer \
    --retrieved2response \
    --retrieved2answer
```


## Computing Metrics

Use the following command for computing the metrics:

```bash
python rag_eval.py --file=<path_to_checking_results_file>
```

## Meta Evaluation



## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.

