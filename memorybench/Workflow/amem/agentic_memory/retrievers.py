import json
import os
from typing import Dict, List, Optional
import ast

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import (
    EmbeddingFunction,
    OpenAIEmbeddingFunction,
)


# ---------------------------------------------------------------------------
# Factory: all embeddings go through OpenAI-compatible API
# ---------------------------------------------------------------------------
def build_embedding_function(
    model_name: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> EmbeddingFunction:
    """
    All embedding models are called via OpenAI-compatible /embeddings API.
    Strip the 'openai:' prefix if present, then call OpenAIEmbeddingFunction.

    Examples:
        'text-embedding-3-small'       -> OpenAI API
        'Qwen/Qwen3-Embedding-8B'      -> your API endpoint
        'openai:my-embed-model'        -> strip prefix, your API endpoint
    """
    if model_name.startswith("openai:"):
        model_name = model_name[len("openai:"):]
    _key = api_key or os.getenv("OPENAI_API_KEY", "")
    _url = base_url or os.getenv("OPENAI_BASE_URL", None)
    kwargs = {"api_key": _key, "model_name": model_name}
    if _url:
        kwargs["api_base"] = _url
    return OpenAIEmbeddingFunction(**kwargs)


# ---------------------------------------------------------------------------
# ChromaRetriever
# ---------------------------------------------------------------------------
class ChromaRetriever:
    """Vector database retrieval using ChromaDB.

    Supports both local (sentence-transformers) and remote (OpenAI-compatible)
    embedding models. Pass api_key / base_url to use remote embeddings.
    """

    def __init__(
        self,
        collection_name: str = "memories",
        model_name: str = "all-MiniLM-L6-v2",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.client = chromadb.Client(Settings(allow_reset=True))
        self.embedding_function = build_embedding_function(model_name, api_key, base_url)
        self.collection = self.client.get_or_create_collection(
            name=collection_name, embedding_function=self.embedding_function
        )

    def add_document(self, document: str, metadata: Dict, doc_id: str):
        processed_metadata = {}
        for key, value in metadata.items():
            if isinstance(value, (list, dict)):
                processed_metadata[key] = json.dumps(value)
            else:
                processed_metadata[key] = str(value)
        self.collection.add(documents=[document], metadatas=[processed_metadata], ids=[doc_id])

    def delete_document(self, doc_id: str):
        self.collection.delete(ids=[doc_id])

    def search(self, query: str, k: int = 5):
        count = self.collection.count()
        if count == 0:
            return {"ids": [[]], "metadatas": [[]], "distances": [[]], "documents": [[]]}
        actual_k = min(k, count)
        results = self.collection.query(query_texts=[query], n_results=actual_k)
        if results and results.get("metadatas"):
            results["metadatas"] = self._convert_metadata_types(results["metadatas"])
        return results

    def _convert_metadata_types(self, metadatas: List[List[Dict]]) -> List[List[Dict]]:
        for query_metadatas in metadatas:
            if isinstance(query_metadatas, list):
                for metadata_dict in query_metadatas:
                    if isinstance(metadata_dict, dict):
                        self._convert_metadata_dict(metadata_dict)
        return metadatas

    def _convert_metadata_dict(self, metadata: Dict) -> None:
        for key, value in metadata.items():
            if not isinstance(value, str):
                continue
            try:
                metadata[key] = ast.literal_eval(value)
            except Exception:
                pass
