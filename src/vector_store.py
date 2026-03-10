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

            # 2. Add extra metadata and filter empty content
            valid_documents = []
            for doc in documents:
                if doc.page_content and doc.page_content.strip():
                    doc.metadata["doc_type"] = doc_type
                    # Lightweight keyword-based tagging (trial)
                    fname = os.path.basename(file_path)
                    low = fname.lower()
                    cn = fname
                    # env
                    if "深海" in cn or "deep" in low:
                        doc.metadata["env"] = "深海"
                    elif "浅海" in cn or "shallow" in low:
                        doc.metadata["env"] = "浅海"
                    elif "港湾" in cn or "近岸" in cn or "harbor" in low:
                        doc.metadata["env"] = "港湾"
                    elif "冰下" in cn or "ice" in low:
                        doc.metadata["env"] = "冰下"
                    # device
                    if "主动" in cn or "active" in low:
                        doc.metadata["device"] = "主动"
                    elif "被动" in cn or "passive" in low:
                        doc.metadata["device"] = "被动"
                    # band
                    if "低频" in cn or "low-frequency" in low:
                        doc.metadata["band"] = "低频"
                    elif "中频" in cn or "mid-frequency" in low or "中频段" in cn:
                        doc.metadata["band"] = "中频"
                    elif "高频" in cn or "high-frequency" in low:
                        doc.metadata["band"] = "高频"
                    # ssp_type
                    if "汇聚区" in cn or "sofar" in low or "声道" in cn:
                        doc.metadata["ssp_type"] = "汇聚区"
                    elif "表面声道" in cn:
                        doc.metadata["ssp_type"] = "表面声道"
                    elif "中层极小" in cn:
                        doc.metadata["ssp_type"] = "中层极小"
                    # bottom_type
                    if "泥" in cn or "mud" in low:
                        doc.metadata["bottom_type"] = "泥"
                    elif "砂" in cn or "sand" in low:
                        doc.metadata["bottom_type"] = "砂"
                    elif "岩" in cn or "rock" in low:
                        doc.metadata["bottom_type"] = "岩"
                    # task
                    if "侦察" in cn or "recon" in low:
                        doc.metadata["task"] = "侦察"
                    elif "跟踪" in cn or "track" in low:
                        doc.metadata["task"] = "跟踪"
                    elif "定位" in cn or "locat" in low:
                        doc.metadata["task"] = "定位"
                    elif "通信" in cn or "commun" in low:
                        doc.metadata["task"] = "通信"
                    # array_type
                    if "线阵" in cn or "line array" in low:
                        doc.metadata["array_type"] = "线阵"
                    elif "面阵" in cn or "planar array" in low:
                        doc.metadata["array_type"] = "面阵"
                    elif "拖曳阵" in cn or "towed array" in low:
                        doc.metadata["array_type"] = "拖曳阵"
                    valid_documents.append(doc)
            
            if not valid_documents:
                return False, "No valid text content after filtering", 0

            # 3. Add to ChromaDB
            self.vectordb.add_documents(valid_documents)
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
        
        valid_exts = ['.docx', '.pdf', '.txt']
        
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
