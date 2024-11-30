import json
from datasets import load_dataset


def process_fiqa():
    print("Processing FIQA data...")
    
    raw_queries = load_dataset("BeIR/fiqa", 'queries')['queries']
    q2gt = json.load(open('metadata/fiqa_metadata.json'))
    
    queries = []
    for q in raw_queries:
        if str(q['_id']) in q2gt:
            queries.append({
                'query_id': str(q['_id']),
                'query': q['text'],
                'gt_answer': q2gt[q['_id']]
            })
    print(f"Number of queries: {len(queries)}")
    json.dump({'input_data': queries}, open('processed_data/fiqa/fiqa_queries.json', 'w'), indent=2)
    
    corpus = load_dataset("BeIR/fiqa", 'corpus')['corpus']
    documents = [{'doc_id': str(doc['_id']), 'title': doc['title'], 'text': doc['text']} for doc in corpus]
    print(f"Number of documents: {len(documents)}")
    with open('processed_data/fiqa/corpus.jsonl', 'w') as f:
        for doc in documents:
            f.write(json.dumps(doc) + '\n')
    print('Documents saved to processed_data/fiqa/corpus.jsonl')
    

if __name__ == "__main__":
    process_fiqa()