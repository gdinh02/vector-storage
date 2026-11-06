import chromadb
from vectordb.embedder import PromptedLlamaCppEmbeddingFunction
from vectordb.pipeline import VectorDatabasePipeline
from vectordb.storage import DirectoryStorage

from pprint import pprint
from dotenv import load_dotenv
import os

if __name__ == "__main__":
    
    load_dotenv()
    
    api_key = os.getenv("API_KEY")
    api_url = os.getenv("API_URL")
    db_url = os.getenv("DB_URL")
    test_docs_dir = os.getenv("TEST_DOCS_DIR")
    
    if None in [api_key, api_url, db_url, test_docs_dir]:
        exit(-1)
    
    chroma_client = chromadb.PersistentClient(path=db_url)
    custom_ef = PromptedLlamaCppEmbeddingFunction(
            api_key=api_key,
            api_base=api_url,
            model_name="EmbeddingGemma2",
            doc_template="title: none | text: {text}",
            query_template="task: search result | query: {text}",
            support_images=True
        )
    storage = DirectoryStorage(test_docs_dir)

    pipeline = VectorDatabasePipeline(
        chromadb=chroma_client,
        embedding_func=custom_ef,
        embedding_batch_size=8,
        file_storage=storage
    )

    pipeline.add_dir_to_collection(
        "testing",
        "."
    )

    results = pipeline.query_database(
        collection="testing",
        queries=["PC Case"]
    )

    pprint(results)