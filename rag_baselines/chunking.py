import argparse
import json
import sys
# avoid llama_index recursion limit error
sys.setrecursionlimit(10000)
import multiprocessing as mp

from tqdm import tqdm
from transformers import AutoTokenizer
from llama_index.core.node_parser import SentenceSplitter


def process_line(line):
    results = []
    item = json.loads(line)
    chunks = text_splitter.split_text(item["text"])
    for i, chunk in enumerate(chunks):
        text = f"{item['title']}. {chunk}" if item["title"] else chunk
        results.append({
            "doc_id": item["doc_id"],
            "title": item["title"],
            "chunk_id": i,
            "text": text
        })
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk_size", type=int, default=300)
    parser.add_argument("--overlap_ratio", type=float, default=0.2)
    parser.add_argument("--num_workers", type=int, default=mp.cpu_count())
    parser.add_argument("--corpus_dir", type=str, default="./ragchecker_corpus")
    parser.add_argument("--data_names", nargs="+", default=[
            "kiwi", "finance", "novelqa", "lifestyle", "recreation",
            "science", "technology", "writing", "bioasq", "clapnq"
    ])
    args = parser.parse_args()
    print(f"Run chunking for {args.data_names}")
    print(
        f"Chunk size: {args.chunk_size} \n"
        f"Overlap: {int(args.overlap_ratio * args.chunk_size)}"
    )

    # using e5-mistral tokenizer for unified tokenization
    tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-mistral-7b-instruct")
    def tokenize_fn(text):
        return tokenizer.tokenize(text)
    
    # initialize sentence splitter
    text_splitter = SentenceSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap=int(args.chunk_size * args.overlap_ratio),
        tokenizer=tokenize_fn,
    )

    for data_name in args.data_names:
        print(f"Chunking {data_name} corpus")
        if args.chunk_size != 300 or args.overlap_ratio != 0.2:
            out_path = f"{args.corpus_dir}/{data_name}/chunks_{args.chunk_size}_{args.overlap_ratio}.jsonl"
        else:
            out_path = f"{args.corpus_dir}/{data_name}/chunks.jsonl"
        with open(f"{args.corpus_dir}/{data_name}/corpus.jsonl") as fin, \
            open(out_path, "w") as fout:
            # multi-process chunking
            with mp.Pool(args.num_workers) as pool:
                for chunks in tqdm(pool.imap_unordered(
                    process_line, fin, chunksize=10
                )):
                    for chunk in chunks:
                        fout.write(json.dumps(chunk) + "\n")
