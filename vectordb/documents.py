from abc import ABC, abstractmethod
import os
import pymupdf
from pathlib import Path
import chromadb.api.types as chromadb_types
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import Iterator

from .storage import Storage

from PIL import Image as PIL_Image
import numpy as np
import mimetypes


class EmbeddableDocument(ABC):
    @abstractmethod
    def to_vectordoc(self) -> Iterator[tuple[chromadb_types.Image | chromadb_types.Document, dict]]:
        """Return document list along with its metadata expected by chromadb"""
        pass

    @abstractmethod
    def doc_type(self) -> str:
        pass

class TextDocument(EmbeddableDocument):
    """EmbeddableDocument type for text files"""
    CHUNK_SIZE      = 1000
    CHUNK_OVERLAP   = 200
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    def __init__(self, storage: Storage, dir_name: str, file_name: str):
        self.storage = storage
        self.dir_name = dir_name
        self.file_name = file_name
        self.file_type = "text"
        self.source_path = f"{dir_name}/{file_name}"

    def to_vectordoc(self) -> Iterator[tuple[chromadb_types.Image | chromadb_types.Document, dict]]:
        with self.storage.read(self.dir_name, self.file_name) as f:
            text = f.read().decode("utf-8")

        chunks = self.text_splitter.split_text(text)

        for chunk in chunks:        
            yield chunk, {
                "source_type": self.file_type,
                "source_path": self.source_path,
                "filename": self.file_name,
            }

    def doc_type(self):
        return self.file_type

class ImageDocument(EmbeddableDocument):
    """EmbeddableDocument type for standalone image files."""
    
    def __init__(self, storage: Storage, dir_name: str, file_name: str):
        self.storage = storage
        self.dir_name = dir_name
        self.file_name = file_name
        self.file_type = "image"
        self.source_path = f"{dir_name}/{file_name}"

    def to_vectordoc(self) -> Iterator[tuple[chromadb_types.Image | chromadb_types.Document, dict]]:
        """Passes the image path through intact and attaches metadata."""
        with self.storage.read(self.dir_name, self.file_name) as f:
            image = PIL_Image.open(f)
            # Ensure image data is loaded into memory before the stream closes
            image_np = np.asarray(image, dtype=np.uint8)

        yield image_np, {
            "source_type": self.file_type,
            "source_path": self.source_path,
            "filename": self.file_name
        }

    def doc_type(self):
        return self.file_type


class PDFDocument(EmbeddableDocument):
    """EmbeddableDocument type for PDFs, rasterizing pages into images."""
    
    def __init__(self, storage: Storage, dir_name: str, file_name: str, max_size: int = 2048):
        self.storage = storage
        self.dir_name = dir_name
        self.file_name = file_name
        self.file_type = "pdf"
        self.source_path = f"{dir_name}/{file_name}"
        self.max_size = max_size

    def to_vectordoc(self) -> Iterator[tuple[chromadb_types.Image | chromadb_types.Document, dict]]:
        """Rasterizes each PDF page into a PNG and returns the generated image paths."""
        with self.storage.read(self.dir_name, self.file_name) as f:
            pdf_bytes = f.read()

        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        base_filename = Path(self.file_name).stem
        
        try:
            for page_num in range(len(doc)):
                page = doc[page_num]
                rect = page.rect 
                
                # Calculate the scaling factor needed to fit within max_size
                scale_w = self.max_size / rect.width
                scale_h = self.max_size / rect.height
                zoom_factor = min(scale_w, scale_h, 3.0)
                
                # Rasterize page
                mat = pymupdf.Matrix(zoom_factor, zoom_factor)
                pix = page.get_pixmap(matrix=mat)

                img_array = np.frombuffer(pix.samples, dtype=np.uint8)
                img_array = img_array.reshape(pix.height, pix.width, pix.n)
                
                yield img_array, {
                    "source_type": "pdf",
                    "source_path": self.source_path,
                    "filename": base_filename,
                    
                    "page_number": page_num + 1
                }
        finally:
            doc.close()

    def doc_type(self):
        return self.file_type


def document_from_storage(storage: Storage, dir_name: str, file_name: str) -> EmbeddableDocument:
    """Factory to generate the correct document type using Storage."""

    mime_type, _ = mimetypes.guess_type(file_name)
    ext = Path(file_name).suffix.lower()

    if (mime_type and mime_type.startswith('text')) or ext in ['.txt', '.md', '.csv']:
        return TextDocument(storage, dir_name, file_name)
    elif ext == '.pdf':
        return PDFDocument(storage, dir_name, file_name)
    elif (mime_type and mime_type.startswith('image')) or ext in ['.jpg', '.jpeg', '.png']:
        return ImageDocument(storage, dir_name, file_name)
    else:
        print(f"Unsupported file type for {file_name}. Skipping.")
        return None

