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
            # 1. Get text chunks
            chunks = doc_processor.get_chunks(file_path)
            if not chunks:
                return False, "No text extracted from document", 0

            # 2. Create Document objects with metadata
            documents = []
            file_name = os.path.basename(file_path)
            for chunk in chunks:
                metadata = {
                    "source": file_name,
                    "doc_type": doc_type
                }
                documents.append(Document(page_content=chunk, metadata=metadata))

            # 3. Add to ChromaDB
            self.vectordb.add_documents(documents)
            # Persist is automatic in newer Chroma versions, but good to know
            
            logger.info(f"Added {len(documents)} chunks to Vector Store")
            return True, f"Successfully added {file_name}", len(documents)

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

# Singleton
vector_store = VectorStoreHandler()
