import os
import json
from typing import List
from opensearchpy import (
    OpenSearch,
    ConnectionTimeout,
)
from tqdm.auto import trange
from embedding import (
    HuggingFaceTextEmbeddingModel,
    BedrockTextEmbeddingModelAPI,
    BGEEmbeddingModel,
    AOSNeuralSparseEmbeddingModel,
)


INDEX_PROPERTIES = {
    "bm25": {"type": "text", "dim": None, "model_name": None},
    "e5_mistral": {
        "type": "vector",
        "dim": 4096,
        "model_name": "intfloat/e5-mistral-7b-instruct"
    },
    "cohere": {
        "type": "vector",
        "dim": 1024,
        "model_name": "cohere.embed-english-v3"},
    "aos_neural_sparse": {
        "type": "sparse_vector",
        "dim": None,
        "model_name": None
    },
}


class OpenSearchClient:
    def __init__(self, config) -> None:
        """
        Initialize the OpenSearch client.
        """
        self.config = config
        # Initialize OpenSearch Client TODO: remove hardcoded values
        self.client = OpenSearch(
            hosts = [{
                "host": os.environ["OPENSEARCH_HOST"],
                "port": os.environ["OPENSEARCH_PORT"],
            }],
            http_auth=(os.environ["OPENSEARCH_USER"], os.environ["OPENSEARCH_PASSWORD"]),
            use_ssl=True,
            verify_certs=False,
            ssl_assert_hostname=False,
            ssl_show_warn=False,
        )
        self.index_type = INDEX_PROPERTIES[config["retriever"]]["type"]
        self.emb_dim = INDEX_PROPERTIES[config["retriever"]]["dim"]
    
    def load_encoder(self, gpu_id=0) -> None:
        if self.config["retriever"] == "e5_mistral":
            self.encoder = HuggingFaceTextEmbeddingModel(
                "intfloat/e5-mistral-7b-instruct", gpu_id
            )
        elif self.config["retriever"] == "cohere":
            self.encoder = BedrockTextEmbeddingModelAPI(
                model_identifier="cohere.embed-english-v3"
            )
        elif self.config["retriever"] == "aos_neural_sparse":
            self.encoder = AOSNeuralSparseEmbeddingModel(
                gpu_id=gpu_id
            )
        elif self.config["retriever"] == "bm25":
            self.encoder = None
        else:
            raise ValueError("Invalid retriever config.")

    def create_index(self) -> None:
        if self.index_type == "vector":
            body = {
                "settings": {
                    "number_of_shards": 1,
                    "number_of_replicas": 1,
                    "index": {
                        "knn": True, 
                        "knn.algo_param.ef_search": 512
                    }
                },
                "mappings": {
                    "properties": {
                        "embedding": {
                            "type": "knn_vector",
                            "dimension": self.emb_dim,
                            "method": {
                                "name": "hnsw",
                                "space_type": "cosinesimil",
                                "engine": "nmslib",
                                "parameters": {
                                    "ef_construction": 512,
                                    "m": 16
                                }
                            },
                        }
                    }
                },
            }
        elif self.index_type == "text":
            body = {
                "mappings": {
                    "properties": {
                        "text": {
                            "type": "text",
                        }
                    }
                }
            }
        else:
            raise ValueError(f"Invalid index type: {self.index_type}")
        keyword_field = self.config["keyword_field"]
        index_name = self.config["index_name"]
        if keyword_field:
            body["mappings"]["properties"][keyword_field] = {
                "type": "keyword",
            }
        self.delete_index()
        self.client.indices.create(index=index_name, body=body)

    def build_index(self, chunks: List):
        """Batch ingest files to the OpenSearch index."""
        batch_size = self.config['batch_size']
        num_chunk = len(chunks)
        for s in trange(0, num_chunk, batch_size, leave=False):
            chunks_batch = chunks[s: s + batch_size]
            self.index_chunks(chunks_batch)

    def index_chunks(self, chunks: list) -> None:
        # Generate index bulk
        bulk = str()
        if self.index_type == "vector":
            texts = [chunk["text"] for chunk in chunks]
            embeddings = self.encoder.get_batch_embeddings(texts, is_query=False)
            for chunk, embedding in zip(chunks, embeddings):
                chunk["embedding"] = embedding
        for chunk in chunks:
            index_id = f"{chunk['doc_id']}-{chunk['chunk_id']}"
            bulk += json.dumps({
                "index": {
                    "_index": self.config["index_name"], 
                    "_id": index_id
                }
            })
            bulk += "\n"
            bulk += json.dumps(chunk)
            bulk += "\n"

        # Keep indexing chunks until it is successful in case of timeout.
        success = False
        while not success:
            try:
                response = self.client.bulk(bulk)
                success = True
            except ConnectionTimeout:
                continue
        if response["errors"]:
            raise RuntimeError(f"Indexing failed with {response['items']['index']['errors']['reason']}")

    def delete_index(self):
        self.client.indices.delete(
            index=self.config["index_name"],
            ignore_unavailable=True
        )

    def get_bool_rank_features(self, embedding):
        clause_list = []
        for token,weight in embedding.items():
            clause_list.append({
                "rank_feature": {
                    "field": f"embedding.{token}",
                    "boost": weight,
                    "linear": {}
                }
            })
        return {
            "bool":{
                "should":clause_list
            }
        }

    def get_search_body(self, query, top_k) -> dict:
        if self.index_type == "sparse_vector":
            assert isinstance(query, dict)
            body = {
                "size": top_k,
                "query": self.get_bool_rank_features(query)
            }
            return body
        if self.index_type == "vector":
            assert isinstance(query, list)
            body = {
                "size": top_k,
                "query": {
                    "knn": {
                        "embedding": {"vector": query, "k": top_k},
                    }
                },
            }
            return body
        if self.index_type == "text":
            body = {
                "size": top_k,
                "query": {
                    "match": {
                        "text": query
                    }
                }
            }
            return body
        raise ValueError(f"Invalid index type: {self.index_type}")

    def query(self, query: str, k: int = None):
        if self.index_type != "text":
            # Get the embedding of the query
            query = self.encoder.get_embedding(query, is_query=True)
        body = self.get_search_body(query, k)
        index_name = self.config["index_name"]
        results = self.client.search(index=index_name, body=body)

        hits = []
        for hit in results["hits"]["hits"]:
            hits.append(
                {
                    "doc_id": hit["_source"]["doc_id"],
                    "chunk_id": hit["_source"]["chunk_id"],
                    "score": hit["_score"],
                    "text": hit["_source"]["text"],
                    "title": hit["_source"]["title"],
                }
            )
        hits = sorted(hits, key=lambda x: x["score"], reverse=True)
        return hits
