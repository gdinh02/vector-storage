from sentence_transformers import SentenceTransformer
from vectordb.documents import document_from_storage
from vectordb.embedder import PromptedLlamaCppEmbeddingFunction
import chromadb
import sqlite3
from pathlib import Path
import hashlib

from pprint import pprint
from itertools import batched

BATCH_SIZE = 8

if __name__ == "__main__":
    chroma_client = chromadb.PersistentClient(path="./data/chroma.db")
    sqlite_client = sqlite3.connect("./data/sqlite.db")

    custom_ef = PromptedLlamaCppEmbeddingFunction(
        api_key="None",
        api_base="http://192.168.0.200:7777/v1",
        model_name="EmbeddingGemma2",
        doc_template="title: none | text: {text}",
        query_template="task: search result | query: {text}",
        support_images=True
    )

    collection = chroma_client.get_or_create_collection(
        name="testing",
        embedding_function=custom_ef
    )

    for file_path in Path("./test-docs").iterdir():
        if file_path.is_file():
            source_document = document_from_storage(file_path)

            if not source_document:
                continue

            with open(file_path, "rb") as f:
                file_hash = hashlib.file_digest(f, "sha256").hexdigest()

            source_type = source_document.doc_type()

            for batch_index, batch in enumerate(batched(source_document.to_vectordoc(), BATCH_SIZE)):
                id_list = []
                embeddable_list = []
                metadata_list = []

                for item_index, (embeddable, metadata) in enumerate(batch):
                    id_list.append(hashlib.sha256(
                        (file_hash + str(BATCH_SIZE * batch_index + item_index)).encode("utf-8")
                    ).hexdigest())

                    embeddable_list.append(embeddable)
                    metadata_list.append(metadata)

                if source_type == "text":
                    collection.upsert(
                        ids=id_list,
                        metadatas=metadata_list,
                        documents=embeddable_list
                    )
                elif source_type in ["pdf", "image"]:
                    collection.upsert(
                        ids=id_list,
                        metadatas=metadata_list,
                        images=embeddable_list
                    )
                    
    query_texts=["Lan Hoang's Driver License"]

    query_embeddings = custom_ef(query_texts)

    results = collection.query(
        query_embeddings=query_embeddings,
        n_results=5
    )

    pprint(results)