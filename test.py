import chromadb
from vectordb.embedder import PromptedLlamaCppEmbeddingFunction
from vectordb.pipeline import VectorDatabasePipeline
from vectordb.storage import DirectoryStorage

from pprint import pprint

if __name__ == "__main__":
    chroma_client = chromadb.PersistentClient(path="./data/chroma.db")
    custom_ef = PromptedLlamaCppEmbeddingFunction(
            api_key="None",
            api_base="http://192.168.0.200:7777/v1",
            model_name="EmbeddingGemma2",
            doc_template="title: none | text: {text}",
            query_template="task: search result | query: {text}",
            support_images=True
        )
    storage = DirectoryStorage("./test-docs")

    pipeline = VectorDatabasePipeline(
        chromadb=chroma_client,
        embedding_func=custom_ef,
        embedding_batch_size=8,
        file_storage=storage
    )

    pipeline.sync_directory(
        "testing",
        "."
    )

    results = pipeline.query_database(
        collection="testing",
        queries=["Election Enrolment Form"]
    )

    pprint(results)