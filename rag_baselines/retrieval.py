import argparse
from opensearch_client import OpenSearchClient
import os
import json
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_names", nargs="+", default=[
        "kiwi", "finance", "novelqa", "lifestyle", "recreation",
        "science", "technology", "writing", "bioasq", "clapnq"
    ])
    parser.add_argument("--query_dir", type=str, default="./ragchecker_queries")
    parser.add_argument("--out_dir", type=str, default="./retrieval_out")
    parser.add_argument("--chunk_size", type=int, default=300)
    parser.add_argument("--overlap_ratio", type=float, default=0.2)
    parser.add_argument(
        "--retriever", type=str, default="bm25",
        choices=["bm25", "e5_mistral", "cohere", "aos_neural_sparse"]
    )
    parser.add_argument("--top_k", type=int, default=20)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    for data_name in args.data_names:
        print("Retrieving data: ", data_name)
        config = {
            "index_name": f"{data_name}_{args.retriever}_{args.chunk_size}_{args.overlap_ratio}",
            "retriever": args.retriever,
        }
        if data_name == "novelqa":
            config["keyword_field"] = "title"
        else:
            config["keyword_field"] = None
        client = OpenSearchClient(config)
        client.load_encoder()
        with open(os.path.join(args.query_dir, f"{data_name}.json")) as fin:
            data = json.load(fin)
            queries = data["input_data"]
        for query in tqdm(queries):
            hits = client.query(query["query"], args.top_k)
            query["hits"] = hits
        out_path = os.path.join(
            args.out_dir,
            f"{data_name}_{args.retriever}_{args.chunk_size}_{args.overlap_ratio}_k{args.top_k}.json"
        )
        with open(out_path, "w") as fout:
            json.dump(data, fout, indent=2)


if __name__ == "__main__":
    main()
