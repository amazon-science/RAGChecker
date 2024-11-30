import json
from datasets import load_dataset


def process_kiwi():
    print('Processing KIWI dataset...')
    
    question_ids = json.load(open('metadata/kiwi_question_ids.json'))
    data = load_dataset('fangyuan/kiwi', split='train')
    
    queries= []
    for _i, qid in enumerate(question_ids):
        example = data[qid]

        last_turn = example['interaction'][-1]
        queries.append({
            'query_id': str(_i),
            'query': example['original_question'],
            'gt_answer': last_turn['answer_1']
        })

    print(f'Number of queries: {len(queries)}')
    json.dump({'input_data': queries}, open('processed_data/kiwi/kiwi_queries.json', 'w'), indent=2)


if __name__ == "__main__":
    process_kiwi()