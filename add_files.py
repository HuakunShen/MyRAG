from app import upload_files2
import os
from pathlib import Path

book_path = Path("/home/huakun/datasets/books/特种教师")
# paths = [str(p) ]
upload_files2("book", list(book_path.iterdir()))
