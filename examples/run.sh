python -m ragchecker.cli \
    --input_path=examples/checking_inputs.json \
    --output_path=examples/checking_outputs.json \
    --extractor_name=bedrock/meta.llama3-70b-instruct-v1:0 \
    --checker_name=bedrock/meta.llama3-70b-instruct-v1:0 \
    --batch_size_extractor=64 \
    --batch_size_checker=64 \
    --metrics all_metrics
