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
            model="qwen:1.8b-chat",
            temperature=0.1 # Low temperature for factual QA
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
        Format source metadata for display
        """
        if not docs:
            return ""
            
        sources = set()
        for doc in docs:
            source = doc.metadata.get('source', 'Unknown')
            sources.add(source)
            
        return "\n\n(信息来源: " + ", ".join(list(sources)) + ")"

    def answer_question(self, question: str) -> Tuple[str, List[Dict]]:
        """
        Main QA function
        Returns: (Answer string, List of source dicts)
        """
        try:
            # 1. Retrieve relevant documents
            logger.info(f"Searching for: {question}")
            docs = vector_store.search(question, k=3)
            
            if not docs:
                return "未在知识库中找到相关文档。", []

            # 2. Build context
            context = self.format_docs(docs)
            
            # 3. Generate prompt
            prompt_text = self.prompt.format(context=context, question=question)
            
            # 4. Call LLM
            logger.info("Calling LLM...")
            response = self.llm.invoke(prompt_text)
            
            # 5. Supplement sources
            final_answer = response.strip() + self.format_sources(docs)
            
            # Return sources as list of dicts for UI if needed
            source_list = [{"source": d.metadata.get('source'), "content": d.page_content} for d in docs]
            
            return final_answer, source_list

        except Exception as e:
            logger.error(f"Error in QA chain: {e}")
            return f"发生错误: {str(e)}", []

# Singleton
qa_chain = QAChainHandler()
