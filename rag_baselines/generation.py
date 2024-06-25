import argparse
import os
import json
import time
from tqdm import tqdm
from litellm import batch_completion


DEFAULT_INST = "Please answer the given question based on the context."

OPT_INST = "You are an accurate and reliable AI assistant capable of answering questions using external documents. Always be faithful to the provided documents and leverage relevant, accurate information from them as much as possible. Be aware that external documents might contain noisy or factually incorrect data. Apply critical reasoning to discern and use the correct information from the context."

PROMPT = """{instruction}

<context>
{contexts}
</context>

Question: {question}

Please answer the question and tag your answer with <answer></answer>.
"""

DOCUMENT_PROMPT = """
<content>
{content}
</content>
"""

MIXTRAL_PROMPT = """<s>[INST] {instruction}

<context>
{contexts}
</context>

Question: {question}

Please answer the question and tag your answer with <answer></answer>.[/INST]
"""

LLAMA_PROMPT = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{instruction}

<context>
{contexts}
</context>

Question: {question}

Please answer the question and tag your answer with <answer></answer>.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""


PROMPT_MAPPING = {
    "Haiku": PROMPT,
    "Sonnet": PROMPT,
    "Llama3-8B": LLAMA_PROMPT,
    "Llama3-70B": LLAMA_PROMPT,
    "Mixtral-8x7B": MIXTRAL_PROMPT,
    "Mistral-7B": MIXTRAL_PROMPT,
    "GPT-4": PROMPT
}

MODEL_MAPPING = {
    "Haiku": "bedrock/anthropic.claude-3-haiku-20240307-v1:0",
    "Sonnet": "bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
    "Llama3-8B": "bedrock/meta.llama3-8b-instruct-v1:0",
    "Llama3-70B": "bedrock/meta.llama3-70b-instruct-v1:0",
    "Mixtral-8x7B": "bedrock/mistral.mixtral-8x7b-instruct-v0:1",
    "Mistral-7B": "bedrock/mistral.mistral-7b-instruct-v0:2",
    "GPT-4": "openai/gpt-4-turbo"
}


def get_messages(model, example, k=20, opt_prompt=False):
    # sorted_hits = sorted(
    #     example["hits"], key=lambda x: x["score"], reverse=True
    # )
    # hits are sorted after retrieval
    contexts = [
        DOCUMENT_PROMPT.format(content=hit["text"])
        for hit in example["hits"][:k]
    ]
    TEMPLATE = PROMPT_MAPPING[model]
    prompt = TEMPLATE.format(
        contexts="".join(contexts), question=example["query"],
        instruction=OPT_INST if opt_prompt else DEFAULT_INST
    )
    return [{
        "content": prompt,
        "role": "user"
    }]


def call_completion(model, messages):
    while True:
        try:
            responses = batch_completion(
                model,
                messages=messages,
                max_tokens=2048,
                temperature=0.0,
            )
            responses = [
                response['choices'][0]['message']['content']
                for response in responses
            ]
            return responses
        except Exception as e:
            print(str(e))
            time.sleep(10)


def format_response(response):
    if "<answer>" in response:
        response = response.split("<answer>")[1]
    if "</answer>" in response:
        response = response.split("</answer>")[0]
    return response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_names", nargs="+", default=[
        "kiwi", "finance", "novelqa", "lifestyle", "recreation",
        "science", "technology", "writing", "bioasq", "clapnq"
    ])
    parser.add_argument("--retrieval_dir", type=str, default="./retrieval_out")
    parser.add_argument("--out_dir", type=str, default="./generation_out")
    parser.add_argument("--chunk_size", type=int, default=300)
    parser.add_argument("--overlap_ratio", type=float, default=0.2)
    parser.add_argument(
        "--retriever", type=str, default="bm25",
        choices=["bm25", "e5_mistral", "cohere", "aos_neural_sparse"]
    )
    parser.add_argument(
        "--generator", type=str, default="GPT-4",
        choices=["Haiku", "Sonnet", "Llama3-8B", "Llama3-70B", "Mixtral-8x7B", "Mistral-7B", "GPT-4"]
    )
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--generation_k", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--opt_prompt", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    model_identifier = MODEL_MAPPING.get(args.generator, None)
    if model_identifier is None:
        raise NotImplementedError(f"Model {args.generator} not supported")

    for data_name in args.data_names:
        input_path = os.path.join(
            args.retrieval_dir,
            f"{data_name}_{args.retriever}_{args.chunk_size}_{args.overlap_ratio}_k{args.top_k}.json"
        )
        output_path = os.path.join(
            args.out_dir,
            f"{data_name}_{args.retriever}_{args.chunk_size}_{args.overlap_ratio}_k{args.generation_k}_{args.generator}.json"
        )
        if args.opt_prompt:
            output_path = output_path.replace(".json", "_opt.json")
        with open(input_path) as fin:
            data = json.load(fin)
            examples = data["input_data"]
        
        for i in tqdm(range(0, len(examples), args.batch_size)):
            examples_batch = examples[i: i + args.batch_size]
            messages_batch = [
                get_messages(
                    args.generator, example, args.generation_k, args.opt_prompt
                )
                for example in examples_batch
            ]
            responses_batch = call_completion(
                model=model_identifier, messages=messages_batch
            )
            for example, response in zip(examples_batch, responses_batch):
                example["llm_raw_output"] = response
                response = format_response(response)
                retrieved_context = example["hits"][:args.generation_k]
                del example["hits"]
                example["response"] = response
                example["retrieved_context"] = retrieved_context
        with open(output_path, "w") as fout:
            json.dump(data, fout, indent=2)


if __name__ == "__main__":
    main()
