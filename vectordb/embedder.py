from chromadb.api.types import Embeddable, Images, Documents, Embeddings
import chromadb.api.types as chromadb_types
import chromadb.utils.embedding_functions as embedding_functions
from sentence_transformers import SentenceTransformer
from PIL import Image
from typing import cast, Union
import numpy as np
from io import BytesIO
import base64
import torch
import requests

class SentenceTransformerEmbedder(embedding_functions.EmbeddingFunction[Embeddable]):
    def __init__(self, model="google/embeddinggemma-2") -> None:
        self.model = SentenceTransformer(model, device="cpu", model_kwargs= { "torch_dtype": torch.float32 })
        
    @staticmethod
    def name() -> str:
        return "SentenceTransformerEmbedder"
    
    def _encode(self, input: Images | Documents, is_query: bool = False) -> Embeddings:
        if len(input) == 0:
            return []
        
        encode_input = input
        
        if isinstance(input[0], chromadb_types.Document):
            prompt_name = "SearchQuery" if is_query else "Document"
        else:
            encode_input = [Image.fromarray(cast(chromadb_types.Image, img_array)) for img_array in input]
            prompt_name = None
        
        embeddings = self.model.encode(
            inputs=encode_input,
            prompt_name=prompt_name
        )
        
        return embeddings.tolist()
    
    def __call__(self, input: Images | Documents) -> Embeddings:
        return self._encode(input, is_query=False)
    
    def embed_query(self, input: Images | Documents) -> Embeddings:
        return self._encode(input, is_query=True)
    
class PromptedLlamaCppEmbeddingFunction(embedding_functions.EmbeddingFunction):
    def __init__(
        self, 
        api_key: str, 
        model_name: str, 
        api_base: str, 
        doc_template: str = "{text}",
        query_template: str = "{text}",
        support_images: bool = False
    ):
        # Clean up the api_base in case it has a trailing slash
        self.api_base = api_base.rstrip('/')
        self.model_name = model_name
        self.doc_template = doc_template
        self.query_template = query_template
        self.support_images = support_images
        
        # Optional: Set up headers if your API requires the key
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

    def _call_llama_cpp(self, batch_inputs: list) -> Embeddings:
        """Helper method to send the request to llama.cpp and parse the embeddings."""
        endpoint = f"{self.api_base}/embeddings"
        
        response = requests.post(
            endpoint,
            headers=self.headers,
            json={
                "model": self.model_name,
                "input": batch_inputs,
                "encoding_format": "float"
            }
        )
        response.raise_for_status()
        
        items = response.json().get("data", [])
        return [item["embedding"] for item in items]

    def _image_to_base64_string(self, image: Image.Image):
        if image.mode != "RGB":
            image = image.convert("RGB")

        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return img_str

    def _make_llamacpp_content(self, input, content_type: str = "text"):
        if content_type == "text":
            return {
                "content": [
                    {"type": "text", "text": input}
                ]
            }

        if content_type == "image":
            return {
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{input}"}
                    },
                ]
            }

    def __call__(self, input: Union[Documents, Images]) -> Embeddings:
        if len(input) == 0:
            return []

        MAX_BATCH_SIZE = 8
        all_embeddings = []
        
        for i in range(0, len(input), MAX_BATCH_SIZE):
            batch = input[i : i + MAX_BATCH_SIZE]
            batch_inputs = []
            
            # 1. Handle Text
            if isinstance(batch[0], str):
                for text in batch:
                    formatted_text = self.doc_template.format(text=text)
                    batch_inputs.append(self._make_llamacpp_content(formatted_text, "text"))
                
                batch_embeddings = self._call_llama_cpp(batch_inputs)
                all_embeddings.extend(batch_embeddings)
                
            # 2. Handle Images
            elif isinstance(batch[0], np.ndarray):
                if not self.support_images:
                    raise ValueError("Image support is disabled.")
                
                for img_array in batch:
                    img = Image.fromarray(img_array.astype(np.uint8))
                    img_str = self._image_to_base64_string(img)
                    batch_inputs.append(self._make_llamacpp_content(img_str, "image"))
                    
                batch_embeddings = self._call_llama_cpp(batch_inputs)
                all_embeddings.extend(batch_embeddings)
        
        return all_embeddings

    def embed_queries(self, queries: Documents) -> Embeddings:
        """Use this method manually before calling collection.query()"""
        if len(queries) == 0:
            return []
            
        batch_inputs = []
        for text in queries:
            formatted_text = self.query_template.format(text=text)
            batch_inputs.append(self._make_llamacpp_content(formatted_text, "text"))
            
        return self._call_llama_cpp(batch_inputs)