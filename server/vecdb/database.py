from qdrant_client import QdrantClient, models
from fastembed import LateInteractionMultimodalEmbedding
from qdrant_client.conversions import common_types as qdrant_types
from PIL import Image

import ctypes
import gc
import threading
from typing import List, Sequence

# Initialize the client
# client = QdrantClient("localhost", port=6333) # For production
class QdrantMultimodalDB():
    def __init__(self,
        model_name: str = "Qdrant/colmodernvbert",
        db_url: str = None,
        db_path: str = None,
        host: str = None,
        port: int = None,
        models_cache: str = None,
        providers: List[str] = ["CPUExecutionProvider"],
        idle_timeout: float | None = 60.0,
    ):
        if db_url:
            self.client = QdrantClient(location=db_url, prefer_grpc=True)
        elif db_path:
            self.client = QdrantClient(path=db_path, prefer_grpc=True)    
        elif host and port:
            self.client = QdrantClient(host=host, port=port, prefer_grpc=True)
        else:
            raise ValueError("Either db_path or host/port must be provided.")

        # Check if model is supported
        if any(m["model"] == model_name for m in LateInteractionMultimodalEmbedding.list_supported_models()):
            self.model_name = model_name
        else:
            raise ValueError(f"{model_name} is not supported by Qdrant")

        self._models_cache = models_cache
        self._providers = list(providers)
        self.idle_timeout = idle_timeout

        self._model: LateInteractionMultimodalEmbedding | None = None
        self._model_lock = threading.Lock()
        self._unload_timer: threading.Timer | None = None

    def _get_model_locked(self) -> LateInteractionMultimodalEmbedding:
        """Must be called while holding self._model_lock."""
        if self._unload_timer is not None:
            self._unload_timer.cancel()
            self._unload_timer = None

        if self._model is None:
            self._model = LateInteractionMultimodalEmbedding(
                model_name=self.model_name,
                cache_dir=self._models_cache,
                providers=self._providers,
            )
        return self._model

    def _schedule_unload_locked(self) -> None:
        """Must be called while holding self._model_lock."""
        if self._unload_timer is not None:
            self._unload_timer.cancel()
            self._unload_timer = None

        if self.idle_timeout and self.idle_timeout > 0:
            timer = threading.Timer(self.idle_timeout, self.unload_model)
            timer.daemon = True
            timer.start()
            self._unload_timer = timer

    def unload_model(self) -> None:
        """Unloads the ONNX session and releases VRAM and Linux CPU heap memory."""
        with self._model_lock:
            if self._unload_timer is not None:
                self._unload_timer.cancel()
                self._unload_timer = None

            if self._model is not None:
                del self._model
                self._model = None
                gc.collect()
                try:
                    ctypes.CDLL("libc.so.6").malloc_trim(0)
                except OSError:
                    pass

    def get_available_collections(self):
        return self.client.get_collections()

    def create_new_collection(self, collection_name: str):
        if not self.client.create_collection(
            collection_name,
            vectors_config=models.VectorParams(
                size=self.client.get_embedding_size(self.model_name),
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                )
            )
        ):
            raise Exception(f"Failed to create {collection_name}")

    def delete_collection(self, collection_name: str):
        if not self.client.delete_collection(collection_name, timeout=30):
            raise Exception(f"Failed to delete {collection_name}")

    def add_to_collection(self, 
                          collection_name: str, 
                          ids: Sequence[qdrant_types.PointId], 
                          documents: Sequence[str] | Sequence[Image.Image],
                          payload: Sequence[dict],
                          batch_size: int = 4,):

        if not (len(ids) == len(documents) == len(payload)):
            raise ValueError("Inputs length must be the same")

        if len(ids) == 0:
            return

        with self._model_lock:
            model = self._get_model_locked()
            try:
                if isinstance(documents[0], str):
                    doc_vectors = list(model.embed_text(documents, batch_size=batch_size))
                elif isinstance(documents[0], Image.Image):
                    doc_vectors = list(model.embed_image(documents, batch_size=batch_size))
                else:
                    raise TypeError(f"Unsupported document type: {type(documents[0])}")
            finally:
                self._schedule_unload_locked()

        self.client.upload_collection(
            collection_name=collection_name,
            vectors=doc_vectors,
            payload=payload,
            ids=ids,
            batch_size=4
        )

    def query_collection(
        self,
        collection_name: str,
        query: str | Image.Image,
        limit: int = 5,
        query_filter: models.Filter | None = None,
        score_threshold: float | None = None,
        with_payload: bool = True,
    ) -> list[models.ScoredPoint]:

        with self._model_lock:
            try:
                model = self._get_model_locked()
                if isinstance(query, str):
                    query_vector = next(model.embed_text([query])).tolist()
                elif isinstance(query, Image.Image):
                    query_vector = next(model.embed_image([query])).tolist()
                else:
                    raise TypeError(f"Unsupported query type: {type(query)}")
            finally:
                self._schedule_unload_locked()

        response = self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit,
            query_filter=query_filter,
            score_threshold=score_threshold,
            with_payload=with_payload,
        )
        return response.points

    def delete_from_collection(self,
                               collection_name: str,
                               ids: Sequence[qdrant_types.PointId]):

        response = self.client.delete(
            collection_name=collection_name,
            points_selector=ids
        )

        return response