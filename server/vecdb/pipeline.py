import shutil
from uuid6 import uuid7
from pathlib import Path

import pymupdf
from PIL import Image
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import models

from .database import QdrantMultimodalDB


class Pipeline:
    IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    TEXT_EXTS = {".txt", ".md", ".csv", ".json"}
    PDF_EXTS = {".pdf"}

    def __init__(
        self,
        db: QdrantMultimodalDB,
        storage_dir: str | Path,
        # collection_name: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        pdf_dpi: int = 150,
    ):
        self.db = db
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        # self.collection_name = collection_name
        self.pdf_dpi = pdf_dpi
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def _ensure_collection(self, collection: str) -> None:
        existing = {c.name for c in self.db.get_available_collections().collections}
        if collection not in existing:
            self.db.create_new_collection(collection)

    def _stored_path(self, file_id: str, suffix: str) -> Path:
        return self.storage_dir / f"{file_id}{suffix.lower()}"

    def _find_stored_file(self, file_id: str) -> Path | None:
        matches = list(self.storage_dir.glob(f"{file_id}.*"))
        return matches[0] if matches else None

    def _extract_documents(
        self, file_path: Path, file_id: str, extra_metadata: dict | None = None
    ) -> tuple[list[str], list[str] | list[Image.Image], list[dict]]:
        ext = file_path.suffix.lower()
        base_meta = {
            "file_id": file_id,
            "filename": file_path.name,
            "file_path": str(file_path),
            "file_type": ext.lstrip("."),
            **(extra_metadata or {}),
        }

        if ext in self.IMAGE_EXTS:
            img = Image.open(file_path).convert("RGB")
            return [str(uuid7())], [img], [{**base_meta, "page": 1}]

        if ext in self.PDF_EXTS:
            images: list[Image.Image] = []
            ids: list[str] = []
            payloads: list[dict] = []
            with pymupdf.open(file_path) as doc:
                for page_idx, page in enumerate(doc, start=1):
                    pix = page.get_pixmap(dpi=self.pdf_dpi)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    images.append(img)
                    ids.append(str(uuid7()))
                    payloads.append({**base_meta, "page": page_idx})
            return ids, images, payloads

        if ext in self.TEXT_EXTS:
            text = file_path.read_text(encoding="utf-8")
            chunks = self.text_splitter.split_text(text)
            ids = [str(uuid7()) for _ in chunks]
            payloads = [
                {**base_meta, "chunk_index": i, "text": chunk}
                for i, chunk in enumerate(chunks)
            ]
            return ids, chunks, payloads

        raise ValueError(f"Unsupported file extension: {ext}")

    def upload_file(
        self,
        source_path: str | Path,
        collection: str,
        file_id: str | None = None,
        metadata: dict | None = None,
    ) -> str:
        """Copies a file into storage_dir, embeds its contents, and indexes it in Qdrant."""
        self._ensure_collection(collection)

        src = Path(source_path)
        if not src.is_file():
            raise FileNotFoundError(f"Source file not found: {src}")

        file_id = file_id or str(uuid7())
        dest = self._stored_path(file_id, src.suffix)
        shutil.copy2(src, dest)

        try:
            ids, documents, payloads = self._extract_documents(dest, file_id, metadata)
            if ids:
                self.db.add_to_collection(
                    collection_name=collection,
                    ids=ids,
                    documents=documents,
                    payload=payloads,
                )
        except Exception:
            # Roll back the saved file if indexing fails
            dest.unlink(missing_ok=True)
            raise

        return file_id

    def delete_file(self, file_id: str, collection: str) -> None:
        """Removes all Qdrant points for file_id and deletes the stored file from disk."""
        selector = models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="file_id",
                        match=models.MatchValue(value=file_id),
                    )
                ]
            )
        )
        self.db.delete_from_collection(
            collection_name=collection,
            ids=selector,
        )

        stored_file = self._find_stored_file(file_id)
        if stored_file:
            stored_file.unlink(missing_ok=True)

    def update_file(
        self,
        file_id: str,
        new_source_path: str | Path,
        metadata: dict | None = None,
    ) -> str:
        """Replaces an existing file's stored copy and Qdrant vectors under the same file_id."""
        self.delete_file(file_id)
        return self.upload_file(new_source_path, file_id=file_id, metadata=metadata)

    def query(
        self,
        query: str | Image.Image,
        collection: str,
        limit: int = 5,
        file_id: str | None = None,
        score_threshold: float | None = None,
    ) -> list[models.ScoredPoint]:
        """Queries the collection using text or an image, optionally scoped to a single file_id."""
        query_filter = None
        if file_id:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="file_id",
                        match=models.MatchValue(value=file_id),
                    )
                ]
            )

        return self.db.query_collection(
            collection_name=collection,
            query=query,
            limit=limit,
            query_filter=query_filter,
            score_threshold=score_threshold,
        )