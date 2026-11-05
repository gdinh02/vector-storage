from abc import ABC, abstractmethod
from pathlib import Path
from typing import BinaryIO, List, Union
import shutil


class Storage(ABC):
    @abstractmethod
    def read(self, dir: str, file: str) -> BinaryIO:
        """Returns a readable binary stream of the file."""
        pass

    @abstractmethod
    def write(self, dir: str, file: str, content: Union[bytes, str, BinaryIO]):
        """Writes bytes, strings, or a stream to the file."""
        pass

    @abstractmethod
    def list_dir(self, dir: str) -> List[str]:
        """Returns a list of filenames in the directory."""
        pass

class DirectoryStorage(Storage):
    def __init__(self, base: str = ""):
        self.base = base

    def read(self, dir: str, file: str) -> BinaryIO:
        """Reads and returns a binary stream of the file."""
        # Use pathlib's division operator (/) to join paths cleanly.
        # This prevents "base" and "dir" from merging without a slash.
        file_path = Path(self.base) / dir / file
        
        return open(file_path, 'rb')

    def write(self, dir: str, file: str, content: Union[bytes, str, BinaryIO]) -> None:
        """Writes content to the file, creating directories if needed."""
        dir_path = Path(self.base) / dir
        
        dir_path.mkdir(parents=True, exist_ok=True)
        
        file_path = dir_path / file

        with open(file_path, 'wb') as f:
            if isinstance(content, str):
                f.write(content.encode('utf-8'))
            elif isinstance(content, bytes):
                f.write(content)
            else:
                shutil.copyfileobj(content, f)

    def list_dir(self, dir: str) -> List[str]:
        """Lists files in the directory relative to the base path."""
        # Ensure we include the base path when looking up the directory
        dir_path = Path(self.base) / dir
        
        if not dir_path.exists():
            return []
            
        return [item.name for item in dir_path.iterdir() if item.is_file()]