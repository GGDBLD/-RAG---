from typing import Tuple, List, Dict
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from src.vector_store import vector_store
from src.utils import setup_logger
import os
from sentence_transformers import CrossEncoder

logger = setup_logger('qa_chain')

class QAChainHandler:
    def __init__(self):
        # Initialize Reranker
        # Try to load local reranker model, fallback to None if not found
        self.reranker_path = r"e:\rag_project\models\bge-reranker-base"
        self.reranker = None
        if os.path.exists(self.reranker_path):
            try:
                logger.info(f"Loading Reranker model from {self.reranker_path}...")
                self.reranker = CrossEncoder(self.reranker_path)
                logger.info("Reranker loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load Reranker: {e}")
        else:
            logger.warning(f"Reranker model not found at {self.reranker_path}. Running in retrieval-only mode.")

        # Initialize Ollama
        # Assumes Ollama is running at default local address
        self.llm = Ollama(
            base_url="http://127.0.0.1:11434",
            model="qwen:1.8b",
            temperature=0.2,  # Slightly increase to avoid getting stuck
            # Removed repeat_penalty to prevent gibberish output
            top_k=40,
            top_p=0.9
        )
        
        # Define Prompt Template
        template = """你是一个智能助手。根据【已知信息】回答【问题】。
如果【已知信息】中没有答案，请直接说“不知道”。不要编造。

【已知信息】：
{context}

【问题】：
{question}

【回答】："""
        
        self.prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

    def format_docs(self, docs: List[Document]) -> str:
        """
        Format retrieved documents into a context string
        """
        formatted_docs = []
        for i, doc in enumerate(docs):
            content = doc.page_content.replace('\n', ' ')
            formatted_docs.append(f"片段{i+1}: {content}")
        return "\n\n".join(formatted_docs)

    def format_sources(self, docs: List[Document]) -> str:
        """
        Format source metadata for display (Source + Page)
        """
        if not docs:
            return ""
            
        sources = []
        for doc in docs:
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', 'N/A')
            sources.append(f"{source} (Page {page})")
            
        # Deduplicate while preserving order
        seen = set()
        unique_sources = []
        for s in sources:
            if s not in seen:
                unique_sources.append(s)
                seen.add(s)
            
        return "\n\n(信息来源: " + ", ".join(unique_sources) + ")"

    def answer_question(self, question: str, chat_history: List[Tuple[str, str]] = None) -> Tuple[str, List[Dict]]:
        """
        Main QA function
        Args:
            question: Current user question
            chat_history: List of (user_input, ai_response) tuples
        Returns: (Answer string, List of source dicts)
        """
        try:
            # 1. Condense question if history exists
            search_query = question
            if chat_history and len(chat_history) > 0:
                logger.info("Contextualizing question...")
                # Simple prompt to rephrase question based on history
                history_text = ""
                # Handle Gradio 3.x tuple format: [(user, bot), (user, bot)]
                # Or list of lists: [[user, bot], [user, bot]]
                # Or messages format: [{'role': 'user', 'content': ...}, ...]
                # Or ChatMessage objects (Gradio 4.x+)
                for interaction in chat_history[-6:]: # Use last few interactions
                    if isinstance(interaction, dict):
                        role = interaction.get('role', '')
                        content = interaction.get('content', '')
                        if role == 'user':
                             history_text += f"User: {content}\n"
                        elif role == 'assistant':
                             history_text += f"Assistant: {content}\n"
                    elif hasattr(interaction, 'role') and hasattr(interaction, 'content'):
                        # Handle ChatMessage objects
                        role = getattr(interaction, 'role', '')
                        content = getattr(interaction, 'content', '')
                        if role == 'user':
                             history_text += f"User: {content}\n"
                        elif role == 'assistant':
                             history_text += f"Assistant: {content}\n"
                    elif isinstance(interaction, (list, tuple)) and len(interaction) == 2:
                        q, a = interaction[0], interaction[1]
                        history_text += f"User: {q}\nAssistant: {a}\n"
                
                rephrase_prompt = (
                    f"Given the following conversation and a follow up question, "
                    f"rephrase the follow up question to be a standalone question. "
                    f"If the question is already standalone, just return it as is.\n\n"
                    f"Chat History:\n{history_text}\n"
                    f"Follow Up Input: {question}\n"
                    f"Standalone Question:"
                )
                
                try:
                    search_query = self.llm.invoke(rephrase_prompt).strip()
                    logger.info(f"Rephrased: '{question}' -> '{search_query}'")
                except Exception as e:
                    logger.warning(f"Failed to rephrase question: {e}")
                    search_query = question

            # 2. Retrieve documents
            logger.info(f"Retrieving documents for: {search_query}")
            
            # Rerank strategy: Retrieve more candidates first, then rerank
            initial_k = 20 if self.reranker else 3
            candidate_docs = vector_store.search(search_query, k=initial_k)
            
            docs = candidate_docs
            if self.reranker and candidate_docs:
                logger.info("Reranking documents...")
                try:
                    pairs = [[search_query, doc.page_content] for doc in candidate_docs]
                    scores = self.reranker.predict(pairs)
                    scored_docs = list(zip(candidate_docs, scores))
                    scored_docs.sort(key=lambda x: x[1], reverse=True)
                    docs = [doc for doc, score in scored_docs[:3]]
                    
                    print("\n--- Rerank Results ---")
                    for i, (doc, score) in enumerate(scored_docs[:3]):
                        print(f"[{i+1}] Score: {score:.4f} | Content: {doc.page_content[:50]}...")
                    print("----------------------\n")
                except Exception as e:
                    logger.error(f"Reranking failed: {e}. Fallback to original order.")
                    docs = candidate_docs[:3]
            
            # DEBUG: Print context to console
            print(f"\n--- DEBUG: Context for '{search_query}' ---")
            for i, doc in enumerate(docs):
                snippet = doc.page_content.replace('\n', ' ')[:200]
                print(f"[Doc {i}] {snippet}...")
            print("-------------------------------------------\n")

            if not docs:
                return "抱歉，知识库中没有找到相关信息。", []

            # 3. Generate Answer
            context = self.format_docs(docs)
            chain = self.prompt | self.llm
            
            logger.info("Generating answer...")
            response = chain.invoke({
                "context": context,
                "question": question
            })
            
            # Post-processing: Anti-repetition
            # Simple check: if a substring of length > 10 repeats > 3 times, truncate it
            # Or just check if the last 50 chars are identical to previous 50 chars
            final_text = response.strip()
            
            # Basic cleanup for common loop patterns
            lines = final_text.split('\n')
            unique_lines = []
            seen_lines = set()
            for line in lines:
                if len(line) > 10 and line in seen_lines:
                    continue # Skip duplicate long lines
                unique_lines.append(line)
                seen_lines.add(line)
            final_text = "\n".join(unique_lines)

            # 6. Supplement sources
            final_answer = final_text + self.format_sources(docs)
            
            # Return sources as list of dicts for UI if needed
            source_list = [{"source": d.metadata.get('source'), "page": d.metadata.get('page'), "content": d.page_content} for d in docs]
            
            return final_answer, source_list

        except Exception as e:
            logger.error(f"Error in QA chain: {e}")
            return f"发生错误: {str(e)}", []

# Singleton
qa_chain = QAChainHandler()
