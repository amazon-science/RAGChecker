for rg in "bm25 GPT-4" "e5_mistral Llama3-70B" "e5_mistral GPT-4"
do
    set -- $rg # Convert the "tuple" into the param args $1 $2...
    echo $1 and $2
    OPENAI_API_KEY=<your openai key> \
    AWS_REGION_NAME=<aws bedrock region> \
    python generation.py --data_names writing \
        --chunk_size 300 \
        --retriever $1 \
        --generator $2 \
        --top_k 20 \
        --generation_k 20 \
        --retrieval_dir retrieval_out \
        --out_dir generation_out
done
