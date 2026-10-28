import chromadb
from chromadb.api.types import Embeddable, Images, Documents, Embeddings
import chromadb.api.types as chromadb_types
import chromadb.utils.embedding_functions as embedding_functions
from sentence_transformers import SentenceTransformer
from converter import MultimodalIngestor
from pathlib import Path
from pprint import pprint
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
                    batch_inputs.append({
                        "content": [
                            {"type": "text", "text": formatted_text}
                        ]
                    })
                
                batch_embeddings = self._call_llama_cpp(batch_inputs)
                all_embeddings.extend(batch_embeddings)
                
            # 2. Handle Images
            elif isinstance(batch[0], np.ndarray):
                if not self.support_images:
                    raise ValueError("Image support is disabled.")
                
                for img_array in batch:
                    img = Image.fromarray(img_array.astype(np.uint8))
                    
                    buffer = BytesIO()
                    img.save(buffer, format="JPEG", quality=85)
                    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
                    
                    batch_inputs.append({
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}
                            },
                            {"type": "text", "text": "\n"}
                        ]
                    })
                    
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
            batch_inputs.append({
                "content": [
                    {"type": "text", "text": formatted_text}
                ]
            })
            
        return self._call_llama_cpp(batch_inputs)

if __name__ == "__main__":
    client = chromadb.PersistentClient(path="../data/chroma.db")

    # custom_ef = SentenceTransformerEmbedder(
    #     model="google/embeddinggemma-2"
    # )
    
    custom_ef = PromptedLlamaCppEmbeddingFunction(
        api_key="None",
        api_base="http://192.168.0.200:7777/v1",
        model_name="EmbeddingGemma2",
        doc_template="title: none | text: {text}",
        query_template="task: search result | query: {text}",
        support_images=True
    )
    
    collection = client.get_or_create_collection(
        name="testing",
        embedding_function=custom_ef
    )

    # ingestor = MultimodalIngestor(output_dir="../data/pipeline_images")

    # processed_docs = ingestor.process_file("../ga-z270x-gaming5.pdf")

    # img_list = []
    # metadata_list = []
    # id_list = []

    # for doc in processed_docs:
    #     if doc["source_type"] == "pdf_page_image":
    #         doc_image = Image.open(doc["image_path"])
    #         img_array = np.array(doc_image)
    #         doc_metadata = doc["metadata"]
    #         doc_metadata["source_file"] = doc["original_pdf"]

    #         id_list.append(Path(doc["image_path"]).name.split(".")[0])
    #         img_list.append(img_array)
    #         metadata_list.append(doc_metadata)

    # print(len(img_list))
    # print(len(metadata_list))
    # print(len(id_list))
            

    # collection.add(
    #     ids=id_list,
    #     images=img_list,
    #     metadatas=metadata_list
    # )

    query_texts=[
        "PCIe Configurations"
    ]

    results = collection.query(
        query_texts=query_texts,
        n_results=5
    )

    pprint(results)