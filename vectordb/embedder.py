import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer
from converter import MultimodalIngestor
from PIL import Image
from pathlib import Path
from pprint import pprint

class Embedder():
    def __init__(self, model_name: str = "google/embeddinggemma-2"):
        self.model = SentenceTransformer(model_name)

    def encode(self, inputs: list[str | Image.Image], names: list[str]):
        if not isinstance(inputs, list):
            inputs = [inputs]

        if not isinstance(names, list):
            names = names

        if len(inputs) != len(names):
            raise ValueError("input document and name lists have different length")

        items_to_encode = []
        for input_obj, name in zip(inputs, names):
            if not isinstance(input_obj, str):
                item = input_obj
            else:
                item = f"title: {name} | text: {input_obj}"

            items_to_encode.append(item)

        embeddings = self.model.encode(items_to_encode)

        return embeddings.tolist()

    def query(self, queries):
        embeddings = self.model.encode(queries, prompt_name="SearchQuery")

        return embeddings.tolist()


if __name__ == "__main__":
    client = chromadb.PersistentClient(path="../data/chroma.db")

    custom_ef = Embedder(
        model_name="google/embeddinggemma-2"
    )

    collection = client.get_or_create_collection(
        name="testing",
        embedding_function=None
    )

    ingestor = MultimodalIngestor(output_dir="../data/pipeline_images")

    processed_docs = ingestor.process_file("../altera-fifo.pdf")

    embedding_list = []
    metadata_list = []
    id_list = []

    for doc in processed_docs:
        if doc["source_type"] == "pdf_page_image":
            doc_image = Image.open(doc["image_path"])
            doc_metadata = doc["metadata"]
            doc_metadata["source_file"] = doc["original_pdf"]

            id_list.append(Path(doc["image_path"]).name.split(".")[0])
            embedding_list.append(doc_image)
            metadata_list.append(doc_metadata)

    embedding_list = custom_ef.encode(embedding_list, id_list)

    collection.add(
        ids=id_list,
        embeddings=embedding_list,
        metadatas=metadata_list
    )

    query_texts=[
        "Guidelines for Embedded Memory ECC Feature", 
        "DCFIFO Timing Constraint Setting"
    ]
    query_embeddings = custom_ef.query(query_texts)

    results = collection.query(
        query_embeddings=query_embeddings,
        n_results=5
    )

    pprint(results)