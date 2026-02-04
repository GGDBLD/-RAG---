import os
from typing import List, Tuple
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from src.document_processing import doc_processor
from src.utils import setup_logger

logger = setup_logger('vector_store')

class VectorStoreHandler:
    def __init__(self):
        # Initialize Embedding Model
        # Using BAAI/bge-small-zh-v1.5 as requested
        # It will be downloaded to default cache if not present
        logger.info("Initializing Embedding Model...")
        # Use local model path
        model_path = r"e:\rag_project\models\bge-small-zh-v1.5"
        
        try:
            logger.info(f"Loading model from local path: {model_path}")
            self.embedding_function = HuggingFaceEmbeddings(
                model_name=model_path,
                model_kwargs={'device': 'cpu'}, 
                encode_kwargs={'normalize_embeddings': True},
                show_progress=False
            )
        except Exception as e:
            logger.error(f"Failed to load embedding model from {model_path}: {e}")
            raise e
        
        self.persist_directory = "./chroma_db"
        self.collection_name = "water_acoustic_kb"
        
        # Initialize ChromaDB
        logger.info(f"Initializing ChromaDB at {self.persist_directory}")
        self.vectordb = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embedding_function,
            collection_name=self.collection_name
        )

    def add_document(self, file_path: str, doc_type: str) -> Tuple[bool, str, int]:
        """
        Add document to vector store
        Args:
            file_path: Path to the file
            doc_type: 'core' or 'supplement'
        Returns:
            (success, message, num_chunks)
        """
        if not os.path.exists(file_path):
            return False, "File does not exist", 0

        logger.info(f"Processing document: {file_path}")
        try:
            # 1. Get documents with metadata (page, source)
            documents = doc_processor.process(file_path)
            if not documents:
                return False, "No text extracted from document", 0

            # 2. Add extra metadata
            for doc in documents:
                doc.metadata["doc_type"] = doc_type

            # 3. Add to ChromaDB
            self.vectordb.add_documents(documents)
            # Persist is automatic in newer Chroma versions, but good to know
            
            logger.info(f"Added {len(documents)} chunks to Vector Store")
            return True, f"Successfully added {os.path.basename(file_path)}", len(documents)

        except Exception as e:
            logger.error(f"Error adding document: {e}")
            return False, str(e), 0

    def search(self, query: str, k: int = 3) -> List[Document]:
        """
        Search for relevant documents
        """
        try:
            results = self.vectordb.similarity_search(query, k=k)
            return results
        except Exception as e:
            logger.error(f"Error searching: {e}")
            return []

    def get_indexed_files(self) -> List[str]:
        """
        Get list of filenames already indexed in the vector store
        """
        try:
            # Fetch all metadata
            data = self.vectordb.get(include=['metadatas'])
            if not data or 'metadatas' not in data:
                return []
            
            # Extract unique source filenames
            sources = set()
            for meta in data['metadatas']:
                if meta and 'source' in meta:
                    sources.add(meta['source'])
            return list(sources)
        except Exception as e:
            logger.error(f"Error getting indexed files: {e}")
            return []

    def scan_and_ingest(self, folder_path: str) -> List[str]:
        """
        Scan folder for new files and ingest them
        Returns list of newly added files
        """
        if not os.path.exists(folder_path):
            logger.warning(f"Folder not found: {folder_path}")
            return []

        logger.info(f"Scanning folder: {folder_path}")
        indexed_files = self.get_indexed_files()
        logger.info(f"Found {len(indexed_files)} already indexed files.")
        
        added_files = []
        
        # Supported extensions from DocumentProcessor
        valid_exts = ['.docx', '.pdf']
        
        for root, _, files in os.walk(folder_path):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in valid_exts:
                    if file not in indexed_files:
                        full_path = os.path.join(root, file)
                        logger.info(f"Auto-ingesting new file: {file}")
                        # Default to 'core' doc_type for auto-ingested files
                        success, _, _ = self.add_document(full_path, doc_type='core')
                        if success:
                            added_files.append(file)
                    else:
                        logger.debug(f"Skipping already indexed file: {file}")
                        
        if added_files:
            logger.info(f"Auto-ingestion complete. Added {len(added_files)} files: {added_files}")
        else:
            logger.info("No new files found to ingest.")
            
        return added_files

# Singleton
vector_store = VectorStoreHandler()
