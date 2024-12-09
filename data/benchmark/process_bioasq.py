import json
import ijson
from tqdm import tqdm


def process_bioasq():
    print("Processing BioASQ data...")
    bioasq_queries_metadata = json.load(open("metadata/bioasq_queries_metadata.json"))

    queries = []
    raw_data_dir = "raw_data/bioasq/"
    for filename, metadata in bioasq_queries_metadata.items():
        raw_queries = json.load(open(raw_data_dir + filename))["questions"]
        qid_to_query = {q["id"]: q for q in raw_queries}
        for q in metadata:
            gt_answer = None
            if q["ideal_answer_index"] is None:
                gt_answer = qid_to_query[q["id"]]["ideal_answer"]
            else:
                gt_answer = qid_to_query[q["id"]]["ideal_answer"][
                    q["ideal_answer_index"]
                ]
            queries.append(
                {
                    "query_id": q["id"],
                    "query": qid_to_query[q["id"]]["body"],
                    "gt_answer": gt_answer,
                }
            )

    print(f"Number of queries: {len(queries)}")
    json.dump(
        {"input_data": queries},
        open("processed_data/bioasq/bioasq_queries.json", "w"),
        indent=2,
    )

    print("Processing BioASQ corpus, this may take a while...")

    bioasq_doc_indices = json.load(open("metadata/bioasq_doc_indices.json"))
    bioasq_doc_indices = set(bioasq_doc_indices)
    print(f"Number of indices to retrieve: {len(bioasq_doc_indices)}")

    with open(raw_data_dir + "allMeSH_2022.json", "r", encoding="windows-1252") as file:
        parser = ijson.items(file, "articles.item")

        with open("processed_data/bioasq/corpus.jsonl", "w") as output_file:
            for i, d in tqdm(enumerate(parser)):
                if i in bioasq_doc_indices:
                    formatted_article = {
                        "doc_id": str(d["pmid"]),
                        "title": d["title"],
                        "text": d["abstractText"],
                    }
                    output_file.write(json.dumps(formatted_article) + "\n")

    print("Documents saved to processed_data/bioasq/corpus.jsonl")


if __name__ == "__main__":
    process_bioasq()
