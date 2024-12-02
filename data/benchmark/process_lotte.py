import json
import os


def process_lotte():
    domains = ['lifestyle', 'science', 'writing', 'recreation', 'technology']
    lotte_corpus_dir = 'raw_data/lotte'

    for domain in domains:
        print(f'Processing {domain} dataset...')

        forum_id2query = dict()
        with open(f'{lotte_corpus_dir}/{domain}/test/questions.forum.tsv') as f:
            for line in f.readlines():
                qid, query = line.split('\t')
                forum_id2query[str(qid)] = query.strip()
        search_id2query = dict()
        with open(f'{lotte_corpus_dir}/{domain}/test/questions.search.tsv') as f:
            for line in f.readlines():
                qid, query = line.split('\t')
                search_id2query[str(qid)] = query.strip()

        queries = []
        qid2gt = json.load(open(f'metadata/{domain}_qid2gt.json'))
        for qid, gt in qid2gt.items():
            _, source, _, _id = qid.split('-')
            if source == 'forum':
                query = forum_id2query[_id]
            elif source == 'search':
                query = search_id2query[_id]
            queries.append({
                'query_id': qid,
                'query': query,
                'gt_answer': gt
            })
        print(f'Number of queries: {len(queries)}')
        json.dump({'input_data': queries}, open(f'processed_data/{domain}/{domain}_queries.json', 'w'), indent=2)

        docid = 0
        raw_documents = []
        with open(os.path.join(lotte_corpus_dir, domain, 'dev/collection.tsv'), 'r') as f:
            for line in f.readlines():
                doctext = line.split('\t')[1].strip()
                raw_documents.append((str(docid), doctext))
                docid += 1
        with open(os.path.join(lotte_corpus_dir, domain, 'test/collection.tsv'), 'r') as f:
            for line in f.readlines():
                doctext = line.split('\t')[1].strip()
                raw_documents.append((str(docid), doctext))
                docid += 1
        
        doc_indices = json.load(open(f'metadata/{domain}_doc_indices.json'))
        documents = []
        for doc_index in doc_indices:
            docid, doctext = raw_documents[doc_index]
            documents.append({'doc_id': docid, 'text': doctext})
        print(f'Number of documents: {len(documents)}')
        with open(f'processed_data/{domain}/corpus.jsonl', 'w') as f:
            for doc in documents:
                f.write(json.dumps(doc) + '\n')
        print(f'Documents saved to processed_data/{domain}/corpus.jsonl')


if __name__ == "__main__":
    process_lotte()