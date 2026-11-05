from sentence_transformers import SentenceTransformer
from .documents import document_from_storage, EmbeddableDocument
from .embedder import PromptedLlamaCppEmbeddingFunction
from .storage import Storage
import chromadb
from pathlib import Path
import hashlib

from chromadb.utils.embedding_functions import EmbeddingFunction
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
import chromadb.api.types as chromadb_types

from pprint import pprint
from itertools import batched
from io import BufferedReader


class VectorDatabasePipeline:
    def __init__(self,
        chromadb: ClientAPI,
        embedding_func: EmbeddingFunction,
        embedding_batch_size: int,
        file_storage: Storage
    ):
        self.vec_db = chromadb
        self.emb_func = embedding_func
        self.batch_size = embedding_batch_size
        self.file_storage = file_storage

    @staticmethod
    def _get_item_hash(item: object):
        if isinstance(item, str):
            return hashlib.sha256(item).hexdigest()
        elif isinstance(item, BufferedReader):
            return hashlib.file_digest(item, "sha256").hexdigest()
        else:
            raise TypeError(f"Wrong input object type, got: {item.__class__.__name__}")

    def _add_document(self,
        document: EmbeddableDocument,
        collection: Collection,
        base_hash: str
    ):
        source_type = document.doc_type()
        
        for batch_index, batch in enumerate(batched(document.to_vectordoc(), self.batch_size)):
            id_list = []
            embeddable_list = []
            metadata_list = []

            for item_index, (embeddable, metadata) in enumerate(batch):
                id_list.append(hashlib.sha256(
                    (base_hash + str(self.batch_size * batch_index + item_index)).encode("utf-8")
                ).hexdigest())

                embeddable_list.append(embeddable)

                metadata["file_hash"] = base_hash
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

    def add_files_to_collection(self, collection: str, dir_name: str = "", file_names: list[str] = []):
        """Adds files, skipping unmodified ones, and updating modified ones."""
        db_collection = self.vec_db.get_or_create_collection(
            name=collection,
            embedding_function=self.emb_func
        )
        
        for file_name in file_names:
            # Reconstruct the source_path exactly as it is saved by EmbeddableDocument
            source_path = f"{dir_name}/{file_name}"
            
            # 1. Calculate the current hash of the file in storage
            with self.file_storage.read(dir_name, file_name) as f:
                current_file_hash = self._get_item_hash(f)

            # 2. Query ChromaDB to see if this file path is already indexed
            # We MUST include ["metadatas"] this time so we can inspect the stored hash
            existing_records = db_collection.get(
                where={"source_path": source_path},
                limit=1,
                include=["metadatas"] 
            )

            # 3. Handle existence, skipping, and deletion
            is_update = False
            if existing_records and len(existing_records["ids"]) > 0:
                # Extract the hash that was saved previously
                existing_hash = existing_records["metadatas"][0].get("file_hash")
                
                if existing_hash == current_file_hash:
                    print(f"Skipping {file_name}: Unmodified file already exists in database.")
                    continue  # Move to the next file
                else:
                    print(f"Updating {file_name}: Modification detected. Removing old records...")
                    # The file changed. Delete all old chunks associated with this path
                    db_collection.delete(where={"source_path": source_path})
                    is_update = True

            # 4. Instantiate the document parser
            source_document = document_from_storage(
                self.file_storage, 
                dir_name, 
                file_name
            )

            if not source_document:
                continue

            # 5. Add the new or updated document to ChromaDB
            action = "Re-ingesting modified file" if is_update else "Ingesting new file"
            print(f"{action}: {file_name}...")
            
            self._add_document(
                source_document,
                db_collection,
                current_file_hash
            )

    def add_dir_to_collection(self, collection: str, dir_name: str = ""):
        file_names = self.file_storage.list_dir(dir_name)
        self.add_files_to_collection(collection, dir_name, file_names)

    def query_database(self, collection: str, queries: list[str], n_results: int = 5) -> chromadb_types.QueryResult:

        db_collection = self.vec_db.get_collection(
            name=collection,
            embedding_function=self.emb_func
        )
        
        return db_collection.query(query_texts=queries, n_results=n_results)

    def remove_file_by_path(self, collection: str, file_path: str):
        """Deletes all chunks/images associated with a specific file path."""
        db_collection = self.vec_db.get_collection(
            name=collection,
            embedding_function=self.emb_func
        )
        # Delete using the source_path metadata
        db_collection.delete(where={"source_path": file_path})

    def remove_file_by_hash(self, collection: str, file_hash: str):
        """Deletes all chunks/images associated with a specific file hash."""
        db_collection = self.vec_db.get_collection(
            name=collection,
            embedding_function=self.emb_func
        )
        # Delete using the injected file_hash metadata
        db_collection.delete(where={"file_hash": file_hash})

    def list_indexed_files(self, collection: str) -> list[str]:
        """Queries the database to find all unique file paths currently indexed."""
        db_collection = self.vec_db.get_collection(
            name=collection,
            embedding_function=self.emb_func
        )
        
        # Retrieve only metadatas to save memory
        results = db_collection.get(include=["metadatas"])
        
        unique_files = set()
        if results and results["metadatas"]:
            for meta in results["metadatas"]:
                if meta and "source_path" in meta:
                    unique_files.add(meta["source_path"])
                    
        return list(unique_files)

    def sync_directory(self, collection: str, dir_name: str):
        """Synchronizes the database with the storage directory, handling additions, updates, and deletions."""
        print(f"Starting synchronization for directory: '{dir_name}'...")
        
        # 1. Get the current files in storage and format them to match 'source_path'
        storage_file_names = self.file_storage.list_dir(dir_name)
        storage_paths = {f"{dir_name}/{file_name}" for file_name in storage_file_names}

        # 2. Get the currently indexed files from the database
        # (Assuming you added the `list_indexed_files` method from the earlier step)
        all_indexed_paths = set(self.list_indexed_files(collection))
        
        # 3. Filter indexed paths to only those belonging to the current directory
        # This prevents us from accidentally deleting files from other directories 
        # that share the same ChromaDB collection.
        indexed_paths_in_dir = {path for path in all_indexed_paths if path.startswith(f"{dir_name}/")}

        # 4. Find files that exist in the database but are missing from storage
        orphaned_paths = indexed_paths_in_dir - storage_paths

        # 5. Remove orphaned files from the database
        if orphaned_paths:
            db_collection = self.vec_db.get_collection(
                name=collection,
                embedding_function=self.emb_func
            )
            for path in orphaned_paths:
                print(f"Deleting removed file from database: {path}")
                # Use ChromaDB's where filter to delete all chunks associated with this path
                db_collection.delete(where={"source_path": path})
        else:
            print("No deleted files to clean up.")

        # 6. Run the ingest logic to handle new and modified files
        # Since add_files_to_collection already skips unmodified files and updates changed ones,
        # we can just pass it the current storage list.
        self.add_files_to_collection(collection, dir_name, storage_file_names)
        
        print(f"Synchronization complete for '{dir_name}'.")


    
