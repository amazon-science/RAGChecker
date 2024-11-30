import json


def process_bioasq():
    print("Processing BioASQ data...")
    bioasq_queries_metadata = json.load(open('metadata/bioasq_queries_metadata.json'))
    
    queries = []
    raw_data_dir = 'raw_data/bioasq/'
    for filename, metadata in bioasq_queries_metadata.items():
        raw_queries = json.load(open(raw_data_dir + filename))['questions']
        qid_to_query = {q['id']: q for q in raw_queries}
        for q in metadata:
            gt_answer = None
            if q['ideal_answer_index'] is None:
                gt_answer = qid_to_query[q['id']]['ideal_answer']
            else:
                gt_answer = qid_to_query[q['id']]['ideal_answer'][q['ideal_answer_index']]
            queries.append({
                'query_id': q['id'],
                'query': qid_to_query[q['id']]['body'],
                'gt_answer': gt_answer,
            })
    
    print(f"Number of queries: {len(queries)}")
    json.dump({'input_data': queries}, open('processed_data/bioasq/bioasq_queries.json', 'w'), indent=2)

    print('Processing BioASQ corpus, this may take a while...')

    documents = []
    bioasq_doc_indices = json.load(open('metadata/bioasq_doc_indices.json'))
    all_docs = json.load(open(raw_data_dir + 'allMeSH_2022.json', 'r', encoding='windows-1252'))['articles']
    for _i in bioasq_doc_indices:
        d = all_docs[_i]
        documents.append({
            'doc_id': str(d['pmid']),
            'title': d['title'],
            'text': d['abstractText'],
        })
    print(f"Number of documents: {len(documents)}")
    
    with open('processed_data/bioasq/corpus.jsonl', 'w') as f:
        for doc in documents:
            f.write(json.dumps(doc) + '\n')
    print('Documents saved to processed_data/bioasq/corpus.jsonl')


if __name__ == "__main__":
    process_bioasq()