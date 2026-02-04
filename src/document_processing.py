import os
import re
from typing import List
import docx
import fitz  # PyMuPDF
from pypdf import PdfReader
from rapidocr_onnxruntime import RapidOCR
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.utils import setup_logger

logger = setup_logger('document_processing')

class DocumentProcessor:
    def __init__(self):
        # Initialize RapidOCR
        # It's lighter and doesn't have the dependency hell of PaddleOCR
        self.ocr = RapidOCR()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]
        )

    def clean_text(self, text: str) -> str:
        """
        Clean extra spaces and special characters
        """
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        # Remove control characters but keep basic punctuation
        text = text.strip()
        return text

    def process_docx(self, file_path: str) -> str:
        """
        Extract text from Word document
        """
        try:
            doc = docx.Document(file_path)
            full_text = []
            for para in doc.paragraphs:
                full_text.append(para.text)
            return self.clean_text('\n'.join(full_text))
        except Exception as e:
            logger.error(f"Error processing docx {file_path}: {e}")
            return ""

    def process_pdf(self, file_path: str) -> str:
        """
        Extract text from PDF (Text-based or Scanned)
        """
        try:
            reader = PdfReader(file_path)
            raw_text = ""
            
            # First try to extract text directly
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    raw_text += extracted

            # Check if text content is sufficient (>= 10 chars)
            if len(raw_text.strip()) >= 10:
                logger.info(f"PDF {file_path} identified as Text PDF.")
                return self.clean_text(raw_text)
            
            # If content is short, treat as Scanned PDF and use OCR
            logger.info(f"PDF {file_path} identified as Scanned PDF (text len < 10). Using OCR.")
            return self.process_scanned_pdf(file_path)

        except Exception as e:
            logger.error(f"Error processing pdf {file_path}: {e}")
            return ""

    def process_scanned_pdf(self, file_path: str) -> str:
        """
        Use RapidOCR + PyMuPDF to extract text from scanned PDF
        """
        try:
            doc = fitz.open(file_path)
            full_text = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                # Render page to image (zoom=2 for better quality)
                mat = fitz.Matrix(2, 2)
                pix = page.get_pixmap(matrix=mat)
                
                # RapidOCR accepts bytes directly
                img_bytes = pix.tobytes("png")
                
                # result structure: [[[[points], text, score], ...], elapse]
                # Note: rapidocr call returns (result, elapse)
                result, _ = self.ocr(img_bytes)
                
                if result:
                    for line in result:
                        # line structure: [[points], text, score]
                        if line and len(line) >= 2:
                             full_text.append(line[1])
            
            return self.clean_text('\n'.join(full_text))
        except Exception as e:
            logger.error(f"Error in OCR processing for {file_path}: {e}")
            return ""

    def get_chunks(self, file_path: str) -> List[str]:
        """
        Main entry point: process file and return chunks
        """
        file_ext = os.path.splitext(file_path)[1].lower()
        content = ""
        
        if file_ext == '.docx':
            content = self.process_docx(file_path)
        elif file_ext == '.pdf':
            content = self.process_pdf(file_path)
        else:
            logger.warning(f"Unsupported file type: {file_ext}")
            return []

        if not content:
            logger.warning(f"No content extracted from {file_path}")
            return []

        chunks = self.text_splitter.split_text(content)
        logger.info(f"Extracted {len(chunks)} chunks from {file_path}")
        return chunks

# Singleton instance for easy import
doc_processor = DocumentProcessor()
