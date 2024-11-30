import json
import os
from datasets import load_dataset


def parse_documents(file_path, documents, doc_ids):
    with open(file_path, 'r') as f:
        for line in f:
            doc = json.loads(line)
            if doc['example_id'] not in doc_ids:
                documents.append({
                    'doc_id': doc['example_id'],
                    'title': doc['document_title'],
                    'text': doc['document_plaintext']
                })
            doc_ids.add(doc['example_id'])
    return documents, doc_ids


def process_clapnq():
    print('Processing CLAPNQ dataset...')
    
    clapnq_ids = json.load(open('metadata/clapnq_ids.json'))
    clapnq_val = load_dataset('PrimeQA/clapnq', split='validation')
    
    val_id_to_example = {example['id']: example for example in clapnq_val}

    queries = []
    for qid in clapnq_ids:
        example = val_id_to_example[qid]
        queries.append({
            'query_id': example['id'],
            'query': example['input'],
            'gt_answer': example['output'][0]['answer'],
            'gt_context': [{'title': x['title'], 'text': x['text']} for x in example['passages']]
        })
    
    print(f'Number of queries: {len(queries)}')
    
    doc_ids = set()
    documents = []
    
    documents, doc_ids = parse_documents(
        'raw_data/clapnq/original_documents/dev/clapnq_dev_answerable_orig.jsonl', 
        documents, doc_ids)
    documents, doc_ids = parse_documents(
        'raw_data/clapnq/original_documents/dev/clapnq_dev_unanswerable_orig.jsonl', 
        documents, doc_ids)
    documents, doc_ids = parse_documents(
        'raw_data/clapnq/original_documents/train/clapnq_train_answerable_orig.jsonl', 
        documents, doc_ids)
    documents, doc_ids = parse_documents(
        'raw_data/clapnq/original_documents/train/clapnq_train_unanswerable_orig.jsonl', 
        documents, doc_ids)
    
    print(f'Number of documents: {len(documents)}')

    if not os.path.exists('processed_data'):
        os.mkdir('processed_data')
    if not os.path.exists('processed_data/clapnq'):
        os.mkdir('processed_data/clapnq')
    
    json.dump({'input_data': queries}, open('processed_data/clapnq/clapnq_queries.json', 'w'), indent=2)
    print('Queries saved to processed_data/clapnq/clapnq_queries.json')
    
    with open('processed_data/clapnq/corpus.jsonl', 'w') as f:
        for doc in documents:
            f.write(json.dumps(doc) + '\n')
    print('Documents saved to processed_data/clapnq/corpus.jsonl')
    

if __name__ == "__main__":
    process_clapnq()