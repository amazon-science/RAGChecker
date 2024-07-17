from llama_index.core.base.response.schema import RESPONSE_TYPE


def response_to_rag_results(
    query: str, 
    gt_answer: str,
    response_object: RESPONSE_TYPE
) -> dict:
    """
    Convert the response object in LlamaIndex to the format of RAGResult.

    Parameters
    ----------
    query : str
        Query.
    gt_answer : str
        Ground truth answer.
    response : RESPONSE_TYPE
        Response from the query engine.

    Returns
    -------
    dict
        Data format that can be converted to RAGResult.
    """
    retrieved_context = [{"text": n.node.text, 'doc_id': n.id_} for n in response_object.source_nodes]
    result = {
        "query_id": None,
        "query": query,
        "gt_answer": gt_answer,
        "response": response_object.response,
        "retrieved_context": retrieved_context,
    }
    return result
