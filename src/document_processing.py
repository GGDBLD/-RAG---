import os
import re
from typing import List
import docx
import fitz  # PyMuPDF
from pypdf import PdfReader
from rapidocr_onnxruntime import RapidOCR
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.utils import setup_logger

logger = setup_logger('document_processing')

class DocumentProcessor:
    def __init__(self):
        # Initialize RapidOCR
        # It's lighter and doesn't have the dependency hell of PaddleOCR
        self.ocr = RapidOCR()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,  # Increased from 500 to reduce fragmentation
            chunk_overlap=150, # Increased overlap for better continuity
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

    def process(self, file_path: str) -> List[Document]:
        """
        Main entry point for processing documents
        Returns list of Document objects with metadata
        """
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return []
            
        ext = os.path.splitext(file_path)[1].lower()
        file_name = os.path.basename(file_path)
        
        try:
            if ext == '.docx':
                return self.process_docx(file_path, file_name)
            elif ext == '.pdf':
                return self.process_pdf(file_path, file_name)
            else:
                logger.warning(f"Unsupported file type: {ext}")
                return []
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return []

    def process_docx(self, file_path: str, file_name: str) -> List[Document]:
        """
        Extract text from Word document
        """
        try:
            doc = docx.Document(file_path)
            full_text = []
            for para in doc.paragraphs:
                full_text.append(para.text)
            text = self.clean_text('\n'.join(full_text))
            
            chunks = self.text_splitter.split_text(text)
            return [Document(page_content=c, metadata={"source": file_name, "page": 1}) for c in chunks]
        except Exception as e:
            logger.error(f"Error processing docx {file_path}: {e}")
            return []

    def process_pdf(self, file_path: str, file_name: str) -> List[Document]:
        """
        Extract text from PDF (Text-based or Scanned)
        """
        try:
            reader = PdfReader(file_path)
            documents = []
            is_scanned = True
            
            # Check first few pages to decide if scanned
            check_text = ""
            for i in range(min(3, len(reader.pages))):
                check_text += reader.pages[i].extract_text() or ""
            
            if len(check_text.strip()) >= 10:
                is_scanned = False
                logger.info(f"PDF {file_name} identified as Text PDF.")
            else:
                logger.info(f"PDF {file_name} identified as Scanned PDF. Using OCR.")
                return self.process_scanned_pdf(file_path, file_name)

            # Process Text PDF page by page
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    clean_text = self.clean_text(text)
                    page_chunks = self.text_splitter.split_text(clean_text)
                    for chunk in page_chunks:
                        documents.append(Document(
                            page_content=chunk, 
                            metadata={"source": file_name, "page": i + 1}
                        ))
            return documents

        except Exception as e:
            logger.error(f"Error processing pdf {file_path}: {e}")
            return []

    def process_scanned_pdf(self, file_path: str, file_name: str) -> List[Document]:
        """
        Use RapidOCR + PyMuPDF to extract text from scanned PDF
        """
        try:
            doc = fitz.open(file_path)
            documents = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                # Render page to image (zoom=2 for better quality)
                mat = fitz.Matrix(2, 2)
                pix = page.get_pixmap(matrix=mat)
                
                # RapidOCR accepts bytes directly
                img_bytes = pix.tobytes("png")
                
                result, _ = self.ocr(img_bytes)
                
                page_text = ""
                if result:
                    for line in result:
                        if line and len(line) >= 2:
                             page_text += line[1] + "\n"
                
                if page_text:
                    clean_text = self.clean_text(page_text)
                    page_chunks = self.text_splitter.split_text(clean_text)
                    for chunk in page_chunks:
                        documents.append(Document(
                            page_content=chunk, 
                            metadata={"source": file_name, "page": page_num + 1}
                        ))
            
            return documents
            
        except Exception as e:
            logger.error(f"Error processing scanned pdf {file_path}: {e}")
            return []
            
    # Deprecated legacy method
    def get_chunks(self, file_path: str) -> List[str]:
        docs = self.process(file_path)
        return [d.page_content for d in docs]

# Singleton
doc_processor = DocumentProcessor()
