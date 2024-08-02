import os
import argparse

from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context, conditional

from langchain_community.document_loaders import TextLoader
from langchain_aws import ChatBedrock, BedrockEmbeddings


def load_documents(doc_dir):
    documents = []
    for filepath in os.listdir(doc_dir):
        if filepath.endswith('.txt'):
            documents += TextLoader(f'{doc_dir}/{filepath}').load()
    return documents


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_model_id", type=str, required=True, default="meta.llama3-1-70b-instruct-v1:0")
    parser.add_argument("--region_name", type=str, required=True, default="us-west-2")
    parser.add_argument("--embedding_model_id", type=str, required=True, default="cohere.embed-english-v3")
    parser.add_argument("--test_size", type=int, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--doc_dir", type=str, required=True)
    args = parser.parse_args()
    
    documents = load_documents(args.doc_dir)
    
    generator_llm = ChatBedrock(
        model_id=args.llm_model_id, 
        region_name=args.region_name
    )
    critic_llm = ChatBedrock(
        model_id=args.llm_model_id, 
        region_name=args.region_name
    )
    embeddings = BedrockEmbeddings(
        model_id=args.embedding_model_id,
        region_name=args.region_name
    )

    generator = TestsetGenerator.from_langchain(
        generator_llm,
        critic_llm,
        embeddings
    )
    
    # Change resulting question type distribution
    distributions = {
        simple: 0.25,
        multi_context: 0.25,
        reasoning: 0.25,
        conditional: 0.25
    }

    testset = generator.generate_with_langchain_docs(
        documents=documents, 
        test_size=args.test_size, 
        distributions=distributions
    )

    test_df = testset.to_pandas()
    test_df.to_csv(args.output_file)
