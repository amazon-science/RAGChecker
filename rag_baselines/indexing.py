import argparse
from multiprocessing import Pool
from opensearch_client import OpenSearchClient
import numpy as np
import torch
import json


def index_chunks(rank, chunks, config):
    client = OpenSearchClient(config)
    client.load_encoder(rank)
    client.build_index(chunks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_names", nargs="+", default=[
        "kiwi", "finance", "novelqa", "lifestyle", "recreation",
        "science", "technology", "writing", "bioasq", "clapnq"
    ])
    parser.add_argument("--corpus_dir", type=str, default="./ragchecker_corpus")
    parser.add_argument("--chunk_size", type=int, default=300)
    parser.add_argument("--overlap_ratio", type=float, default=0.2)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--retriever", type=str, default="bm25",
        choices=["bm25", "e5_mistral", "cohere", "aos_neural_sparse"]
    )
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    for data_name in args.data_names:
        print("Indexing data: ", data_name)
        config = {
            "index_name": f"{data_name}_{args.retriever}_{args.chunk_size}_{args.overlap_ratio}",
            "retriever": args.retriever,
            "batch_size": args.batch_size,
        }
        if data_name == "novelqa":
            config["keyword_field"] = "title"
        else:
            config["keyword_field"] = None
        client = OpenSearchClient(config)
        client.create_index()

        chunks_path = (
            f"{args.corpus_dir}/{data_name}/chunks_{args.chunk_size}"
            f"_{args.overlap_ratio}.jsonl"
        )
        # default setting
        if args.chunk_size == 300 and args.overlap_ratio == 0.2:
            chunks_path = f"{args.corpus_dir}/{data_name}/chunks.jsonl"
        with open(chunks_path) as f:
            chunks = [json.loads(line) for line in f]
        print(f"Indexing {len(chunks)} chunks with {args.num_workers} workers")
        
        if args.retriever not in ["bm25", "cohere"]:
            assert torch.cuda.device_count() >= args.num_workers
        
        chunks_split = np.array_split(chunks, args.num_workers)
        with Pool(args.num_workers) as p:
            p.starmap(
                index_chunks,
                [(i, chunks_split[i], config) for i in range(args.num_workers)]
            )


if __name__ == "__main__":
    main()
