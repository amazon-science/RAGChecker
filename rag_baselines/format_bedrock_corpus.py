import json
import argparse
import os
from tqdm import tqdm
import multiprocessing as mp


def format_bedrock_corpus(input_path, out_dir):
    with open(input_path) as fp:
        for line in tqdm(fp):
            chunk = json.loads(line)
            chunk_id = f"{chunk['doc_id']}-{chunk['chunk_id']}"
            with open(f"{out_dir}/{chunk_id}.txt", "w") as fout:
                fout.write(chunk["text"])
            if not chunk["title"]:
                continue
            with open(f"{out_dir}/{chunk_id}.txt.metadata.json", "w") as fout:
                json.dump({
                    "metadataAttributes": {"title": chunk["title"]},
                }, fout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_names", nargs="+", default=[
        "kiwi", "finance", "novelqa", "lifestyle", "recreation",
        "science", "technology", "writing", "bioasq", "clapnq"
    ])
    parser.add_argument("--corpus_dir", type=str, default="./ragchecker_corpus")
    args = parser.parse_args()
    for data_name in args.data_names:
        input_path = os.path.join(args.corpus_dir, data_name, "chunks.jsonl")
        out_dir = os.path.join(args.corpus_dir, data_name, "bedrock_kb_corpus")
        os.makedirs(out_dir, exist_ok=True)
        format_bedrock_corpus(input_path, out_dir)
