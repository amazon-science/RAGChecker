# Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
import boto3
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
from typing import List, Dict
import json
import itertools
from FlagEmbedding import BGEM3FlagModel
from huggingface_hub import hf_hub_download


class BedrockTextEmbeddingModelAPI:
    def __init__(
        self,
        model_identifier: str = "cohere.embed-english-v3",
    ):
        self.bedrock_client = boto3.client(
            service_name="bedrock-runtime",
            region_name="us-west-2"
        )

        self.model_identifier = model_identifier


    def get_embedding(self, text: str, is_query: bool = False) -> List[float]:
        """
        Get the embedding.

        Args:
            text: the text to compute the embedding.

        Returns:
            The embedding.
        """
        if is_query:
            input_type = "search_query"
        else:
            input_type = "search_document"
        try:
            modelId = self.model_identifier
            accept = 'application/json'
            contentType = 'application/json'
            if self.model_identifier.startswith("cohere"):
                body = json.dumps({
                    'texts': [text[:2000]],
                    'input_type': input_type,
                    'truncate': 'NONE'
                })
            else:
                body = json.dumps({
                    'inputText': text
                })

            response = self.bedrock_client.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)
            response_obj = response.get('body').read().decode('utf-8')
            json_response_obj = json.loads(response_obj)
            if self.model_identifier.startswith("cohere"):
                return json_response_obj['embeddings'][0]
            else:
                return json_response_obj['embedding']
        
        except Exception as e:
            print("ERROR: chunks skipped")
            print(text)
            print(str(e))
            return None
    

    def get_batch_embeddings(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        """
        Get the embeddings for a batch of texts.

        NOTE: Bedrock Text Embedding Model does not support batch embedding. We
        will use for-loop to compute the embeddings. Need to update this code
        when Bedrock Text Embedding Model supports batch embedding.
        Additionally, we need to add 1 sec sleeptime between calls to avoid the
        rate limit.

        Args:
            texts: the texts to compute the embeddings.

        Returns:
            The embeddings.

        """
        if is_query:
            input_type = "search_query"
        else:
            input_type = "search_document"
        try:
            modelId = self.model_identifier
            accept = 'application/json'
            contentType = 'application/json'

            body = json.dumps({
                'texts': [t[:2000] for t in texts],
                'input_type': input_type,
                'truncate':'END'
            })

            response = self.bedrock_client.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)
            response_obj = response.get('body').read().decode('utf-8')
            json_response_obj = json.loads(response_obj)
            return json_response_obj['embeddings']
        
        except Exception as e:
            print("ERROR: chunks skipped")
            print(str(e))
            print(texts)
            return None


class BGEEmbeddingModel:
    def __init__(self, pretrained_model_name, gpu_id) -> None:
        self.model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

    def get_embedding(self, text, is_query=True):
        embeddings = self.model.encode([text])['dense_vecs']
        embedding = embeddings[0].tolist()
        return embedding

    def get_batch_embeddings(self, texts: List[str], is_query=False) -> List[List[float]]:
        embeddings = self.model.encode(texts)['dense_vecs']
        embeddings = embeddings.tolist()
        return embeddings


class HuggingFaceTextEmbeddingModel:
    def __init__(self, pretrained_model_name: str = 'facebook/contriever-msmarco', gpu_id: int = 0):
        
        self.device = f"cuda:{gpu_id}"
        self.pretrained_model_name = pretrained_model_name
            
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
            self.model = AutoModel.from_pretrained(
                pretrained_model_name, torch_dtype="auto"
            )
        except:
            raise ValueError(f"{pretrained_model_name} is not a valid pretrained model name.")

        if torch.cuda.is_available():
            self.model = self.model.to(self.device)
            

    def mean_pooling(self, token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.0)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]

        return sentence_embeddings
    
    def last_token_pool(self, last_hidden_states, attention_mask):
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
        
    def get_embedding(self, text: str, is_query=False) -> List[float]:
        """
        Get the embedding.

        Args:
            text: the text to compute the embedding.

        Returns:
            The embedding.
        """
        with torch.no_grad():
            inputs = self.tokenizer([text], padding=True, truncation=True, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = inputs.to(self.device)

            outputs = self.model(**inputs)
            if self.pretrained_model_name == "intfloat/e5-mistral-7b-instruct":
                embedding = self.last_token_pool(outputs.last_hidden_state, inputs["attention_mask"])
            else:    
                embedding = self.mean_pooling(outputs[0], inputs["attention_mask"])
            embedding = embedding[0].tolist()
        return embedding

    def get_batch_embeddings(self, texts: List[str], is_query=False) -> List[List[float]]:
        """
        Get the embeddings for a batch of texts.

        Args:
            texts: the texts to compute the embeddings.

        Returns:
            The embeddings.

        """
        with torch.no_grad():
            inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = inputs.to(self.device)

            outputs = self.model(**inputs)
            if self.pretrained_model_name == "intfloat/e5-mistral-7b-instruct":
                embeddings = self.last_token_pool(outputs.last_hidden_state, inputs["attention_mask"])
            else:    
                embeddings = self.mean_pooling(outputs[0], inputs["attention_mask"])
            embeddings = embeddings.tolist()
        return embeddings


class AOSNeuralSparseEmbeddingModel:
    def __init__(self, doc_only: bool = False, gpu_id: int = 0):
        self.device = f"cuda:{gpu_id}"
        self.doc_only = doc_only
        if not doc_only:
            self.pretrained_model_name = "opensearch-project/opensearch-neural-sparse-encoding-v1"
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.pretrained_model_name
            )
        else:
            self.pretrained_model_name = "opensearch-project/opensearch-neural-sparse-encoding-doc-v1"
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.pretrained_model_name
            )
            self.idf = self._get_idf()
        self.model = AutoModelForMaskedLM.from_pretrained(
            self.pretrained_model_name
        )
        self.special_token_ids = [
            self.tokenizer.vocab[token]
            for token in self.tokenizer.special_tokens_map.values()
        ]  # special tokens will be masked out
        # id to token mapping used for sparse vector decoding
        self.id2token = [None] * len(self.tokenizer.vocab)
        for token, _id in self.tokenizer.vocab.items():
            self.id2token[_id] = token
        if torch.cuda.is_available():
            self.model = self.model.to(self.device)
        self.model.eval()

    def _get_idf(self):
        """get idf for weights of query tokens"""
        local_path = hf_hub_download(
            "opensearch-project/opensearch-neural-sparse-encoding-doc-v1",
            "idf.json"
        )
        with open(local_path) as f:
            idf = json.load(f)
        idf_vector = [0] * self.tokenizer.vocab_size
        for token, weight in idf.items():
            _id = self.tokenizer._convert_token_to_id_with_added_voc(token)
            idf_vector[_id] = weight
        return torch.tensor(idf_vector, device=self.device)

    def get_embedding(
        self, text: str, is_query: bool = False
    ) -> Dict[str, float]:
        return self.get_batch_embeddings([text], is_query)[0]

    @torch.no_grad()
    def get_batch_embeddings(
        self, texts: List[str], is_query: bool = False
    ) -> List[Dict[str, float]]:
        inputs = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors='pt',
            return_token_type_ids=False
        )
        if torch.cuda.is_available():
            inputs = inputs.to(self.device)
        
        if is_query and self.doc_only:
            values = torch.zeros(
                len(texts), self.tokenizer.vocab_size, device=self.device
            )
            input_ids = inputs["input_ids"]
            bsz = input_ids.size(0)
            values[torch.arange(bsz), input_ids] = 1
            values = values * self.idf
        else:
            embed = self.model(**inputs)[0]
            # sparsify
            values, _ = torch.max(
                embed * inputs["attention_mask"].unsqueeze(-1), dim=1
            )
            values = torch.log(1 + torch.relu(values))
        values[:, self.special_token_ids] = 0
        # decode sparse vector
        sample_indices, token_indices=torch.nonzero(values, as_tuple=True)
        non_zero_values = values[(sample_indices,token_indices)].tolist()
        num_tokens_per_sample = torch.bincount(sample_indices).cpu().tolist()
        tokens = [self.id2token[_id] for _id in token_indices.tolist()]
        output = []
        end_idxs = list(itertools.accumulate([0] + num_tokens_per_sample))
        for i in range(len(end_idxs)-1):
            token_strings = tokens[end_idxs[i]: end_idxs[i+1]]
            weights = non_zero_values[end_idxs[i]: end_idxs[i+1]]
            output.append(dict(zip(token_strings, weights)))
        assert len(output) == len(texts), "output length mismatch"
        return output


if __name__ == "__main__":
    # test e5 mistral
    model = HuggingFaceTextEmbeddingModel("intfloat/e5-mistral-7b-instruct", 0)
    print(model.model.embed_tokens.weight.dtype)
    text1 = " ".join(["I", "am", "a", "test", "sentence."] * 60)
    text2 = " ".join(["today", "is", "a", "good", "day."] * 60)
    embedding1 = torch.tensor(model.get_embedding(text1))
    embedding2 = torch.tensor(model.get_embedding(text2))
    from torch.nn.functional import cosine_similarity
    print(cosine_similarity(embedding1[None, ...], embedding2[None, ...]))
