for retriever in bm25 e5_mistral
do
    OPENSEARCH_HOST=<your host url> \
    OPENSEARCH_PORT=<your host port> \
    OPENSEARCH_USERNAME=<your username> \
    OPENSEARCH_PASSWORD=<your password> \
    python indexing.py --data_names writing kiwi finance \
        --chunk_size 300 \
        --num_workers 8 \
        --retriever $retriever
done
