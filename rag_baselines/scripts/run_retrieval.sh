for retriever in bm25 e5_mistral
do
    OPENSEARCH_HOST=<your host url> \
    OPENSEARCH_PORT=<your host port> \
    OPENSEARCH_USERNAME=<your username> \
    OPENSEARCH_PASSWORD=<your password> \
    python retrieval.py --data_names writing \
        --chunk_size 300 \
        --retriever $retriever \
        --top_k 20 \
        --out_dir retrieval_out
done
