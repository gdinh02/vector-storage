import os
import mimetypes
import pymupdf
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

class MultimodalIngestor:
    def __init__(self, output_dir="./processed_visuals", chunk_size=1000, chunk_overlap=150):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Configure the text splitter for optimal semantic chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )

    def process_file(self, file_path):
        """Routes the file to the appropriate processing method based on MIME type or extension."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        mime_type, _ = mimetypes.guess_type(file_path)
        ext = Path(file_path).suffix.lower()

        if mime_type and mime_type.startswith('text') or ext in ['.txt', '.md', '.csv']:
            return self._process_text(file_path)
        elif ext == '.pdf':
            return self._process_pdf(file_path)
        elif mime_type and mime_type.startswith('image') or ext in ['.jpg', '.jpeg', '.png']:
            return self._process_image(file_path)
        elif mime_type and mime_type.startswith('video') or ext in ['.mp4', '.avi', '.mov']:
            return self._process_video(file_path)
        else:
            print(f"Unsupported file type for {file_path}. Skipping.")
            return []

    def _process_text(self, file_path):
        """Chunks text files into smaller semantic segments."""
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        chunks = self.text_splitter.split_text(text)
        
        return [{
            "source_type": "text",
            "source_path": file_path,
            "content": chunk,  # Ready directly for embeddinggemma2
            "metadata": {"chunk_index": i}
        } for i, chunk in enumerate(chunks)]

    def _process_pdf(self, file_path, max_size=1024):
        """Rasterizes each PDF page into a high-resolution PNG image."""
        doc = pymupdf.open(file_path)
        base_filename = Path(file_path).stem
        processed_pages = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # # 1. Get the original physical dimensions of the PDF page
            rect = page.rect 
            
            # # 2. Calculate the scaling factor needed to fit within max_size
            scale_w = max_size / rect.width
            scale_h = max_size / rect.height
            
            zoom_factor = min(scale_w, scale_h, 3.0)
            
            # Matrix(2, 2) doubles the resolution (roughly 144 DPI) for better VLM legibility
            mat = pymupdf.Matrix(zoom_factor, zoom_factor)
            pix = page.get_pixmap(matrix=mat)
            
            output_image_path = os.path.join(self.output_dir, f"{base_filename}_page_{page_num + 1}.png")
            pix.save(output_image_path)
            
            processed_pages.append({
                "source_type": "pdf_page_image",
                "original_pdf": file_path,
                "image_path": output_image_path, # Requires VLM processing before embedding
                "metadata": {"page_number": page_num + 1}
            })
            
        doc.close()
        return processed_pages

    def _process_image(self, file_path):
        """Passes images through intact, attaching metadata."""
        return [{
            "source_type": "image",
            "image_path": file_path, # Requires VLM processing before embedding
            "metadata": {"filename": Path(file_path).name}
        }]

    def _process_video(self, file_path):
        """Passes videos through intact, attaching metadata."""
        return [{
            "source_type": "video",
            "video_path": file_path, # Requires Keyframe extraction & VLM before embedding
            "metadata": {"filename": Path(file_path).name}
        }]

# --- Example Usage ---
if __name__ == "__main__":
    ingestor = MultimodalIngestor(output_dir="../data/pipeline_images")
    
    # Example payloads (assuming these files exist locally)
    # results = ingestor.process_file("architecture_spec.txt")
    # results = ingestor.process_file("quarterly_report.pdf")
    # results = ingestor.process_file("diagram.png")
    results = ingestor.process_file("../altera-fifo.pdf")