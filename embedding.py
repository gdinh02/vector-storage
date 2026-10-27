from sentence_transformers import SentenceTransformer
import time
from PIL import Image
from vectordb.converter import Converter
from vectordb.embedder import EmbeddingGemma2Function
import chromadb

if __name__ == "__main__":
    client = chromadb.PersistentClient(path="../data/chroma.db")

    custom_ef = EmbeddingGemma2Function(
        model_name="google/embeddinggemma-2"
    )

    converter = Converter()

    collection = client.get_or_create_collection(
        name="testing",
        embedding_function=custom_ef
    )

    page_images = converter.process("altera-fifo.pdf")

    results = collection.query(
        query_texts=["How do I use custom models?"],
        n_results=1
    )

    print(results)