import json


def process_novelqa():
    print('Processing NovelQA dataset...')
    
    books = [
        "Wuthering Heights",
        "The Old Wives' Tale",
        "The Waves",
        "Ayala's Angel",
        "Winesburg, Ohio",
        "The History of Tom Jones",
        "Can You Forgive Her",
        "Crime And Punishment",
        "Les Miserables",
        "White Fang",
        "Mansfield Park",
        "Marcella",
        "Pride and Prejudice",
        "Dubliners",
        "Lover or Friend",
        "Emma",
        "The History of Rome",
        "Sons and Lovers",
        "Oliver Twist"
    ]
    
    # get books corpus
    documents = []
    for _i, book in enumerate(books):
        with open(f'raw_data/NovelQA/Books/PublicDomain/{book}.txt', 'r') as file:
            documents.append({
                "doc_id": str(_i + 1),
                "text": file.read(),
                "title": book
            })
    print(f'Number of documents: {len(documents)}')
    
    with open('processed_data/novelqa/corpus.jsonl', 'w') as f:
        for doc in documents:
            f.write(json.dumps(doc) + '\n')
    print('Documents saved to processed_data/novelqa/corpus.jsonl')
    


if __name__ == "__main__":
    process_novelqa()