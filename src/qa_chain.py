from typing import Tuple, List, Dict
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from src.vector_store import vector_store
from src.utils import setup_logger

logger = setup_logger('qa_chain')

class QAChainHandler:
    def __init__(self):
        # Initialize Ollama
        # Assumes Ollama is running at default local address
        self.llm = Ollama(
            base_url="http://127.0.0.1:11434",
            model="qwen:1.8b",
            temperature=0.3, # Slightly increased to avoid repetition loops
            repeat_penalty=1.3, # Penalize repetition strongly
            top_k=40,
            top_p=0.9
        )
        
        # Define Prompt Template
        template = """基于以下已知信息，回答用户的问题。
如果无法从已知信息中得到答案，请直接回答“未找到相关答案”，不要编造信息。

已知信息：
{context}

问题：
{question}

回答："""
        
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

            # 2. Retrieve relevant documents
            logger.info(f"Searching for: {search_query}")
            docs = vector_store.search(search_query, k=3)
            
            if not docs:
                return "未在知识库中找到相关文档。", []

            # 3. Build context
            context = self.format_docs(docs)
            
            # 4. Generate prompt
            # Use original question for the answer generation to keep tone, 
            # or use search_query? Usually original question is better for "You asked...", 
            # but search_query is better for context match. 
            # Let's use search_query for clarity in what is being answered.
            prompt_text = self.prompt.format(context=context, question=search_query)
            
            # 5. Call LLM
            logger.info("Calling LLM...")
            response = self.llm.invoke(prompt_text)
            
            # 6. Supplement sources
            final_answer = response.strip() + self.format_sources(docs)
            
            # Return sources as list of dicts for UI if needed
            source_list = [{"source": d.metadata.get('source'), "page": d.metadata.get('page'), "content": d.page_content} for d in docs]
            
            return final_answer, source_list

        except Exception as e:
            logger.error(f"Error in QA chain: {e}")
            return f"发生错误: {str(e)}", []

# Singleton
qa_chain = QAChainHandler()
