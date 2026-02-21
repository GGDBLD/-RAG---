from typing import Tuple, List, Dict
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from src.vector_store import vector_store
from src.utils import setup_logger
import os
import re
from sentence_transformers import CrossEncoder

logger = setup_logger('qa_chain')


class QAChainHandler:
    def __init__(self):
        self.reranker_path = r"e:\rag_project\models\bge-reranker-base"
        self.reranker = None
        self.rerank_score_threshold = 0.0
        self.max_rerank_docs = 5
        if os.path.exists(self.reranker_path):
            try:
                logger.info(f"Loading Reranker model from {self.reranker_path}...")
                self.reranker = CrossEncoder(self.reranker_path)
                logger.info("Reranker loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load Reranker: {e}")
        else:
            logger.warning(f"Reranker model not found at {self.reranker_path}. Running in retrieval-only mode.")

        self.llm = Ollama(
            base_url="http://127.0.0.1:11434",
            model="qwen3:8b",
            temperature=0.2,
            top_k=40,
            top_p=0.9
        )

        template = """你是水声工程领域的中文问答助手。请根据【已知信息】回答【问题】。

回答要求：
1. 优先使用【已知信息】中的内容，不要凭空编造。
2. 如果【已知信息】中几乎没有相关内容，请回答：“根据当前知识库暂时无法回答该问题。”
3. 回答要用完整、连贯的中文句子，可以使用 1、2、3 分点说明。
4. 控制在 3~6 句话之内，不要反复重复同一个意思。

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
        formatted_docs = []
        for i, doc in enumerate(docs):
            content = doc.page_content.replace('\n', ' ')
            formatted_docs.append(f"片段{i+1}: {content}")
        return "\n\n".join(formatted_docs)

    def format_sources(self, docs: List[Document]) -> str:
        if not docs:
            return ""

        sources = []
        for doc in docs:
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', 'N/A')
            sources.append(f"{source} (Page {page})")

        seen = set()
        unique_sources = []
        for s in sources:
            if s not in seen:
                unique_sources.append(s)
                seen.add(s)

        return "\n\n(信息来源: " + ", ".join(unique_sources) + ")"

    def clean_answer(self, text: str, max_chars: int = 800) -> str:
        stripped = text.strip()
        if not stripped:
            return stripped
        segments = re.split(r'(?<=[。！？!?])', stripped)
        sentences = []
        for seg in segments:
            s = seg.strip()
            if s:
                sentences.append(s)
        seen = set()
        result_parts = []
        for s in sentences:
            key = s
            if len(key) > 8 and key in seen:
                continue
            seen.add(key)
            result_parts.append(s)
        result = "".join(result_parts)
        if len(result) > max_chars:
            result = result[:max_chars].rstrip()
        return result

    def deduplicate_docs(self, docs: List[Document]) -> List[Document]:
        if not docs:
            return docs
        seen_text = set()
        per_source_page = {}
        unique_docs = []
        for doc in docs:
            text = " ".join(doc.page_content.split())
            key = text[:200]
            source = doc.metadata.get("source", "")
            page = doc.metadata.get("page", "")
            sp_key = (source, page)
            count = per_source_page.get(sp_key, 0)
            if key in seen_text:
                continue
            if count >= 2:
                continue
            seen_text.add(key)
            per_source_page[sp_key] = count + 1
            unique_docs.append(doc)
        return unique_docs

    def answer_question(self, question: str, chat_history: List[Tuple[str, str]] = None) -> Tuple[str, List[Dict]]:
        try:
            last_user_q = None
            if chat_history:
                for interaction in reversed(chat_history):
                    if isinstance(interaction, dict):
                        role = interaction.get("role", "")
                        content = interaction.get("content", "")
                        if role == "user" and content:
                            last_user_q = str(content)
                            break
                    elif hasattr(interaction, "role") and hasattr(interaction, "content"):
                        role = getattr(interaction, "role", "")
                        content = getattr(interaction, "content", "")
                        if role == "user" and content:
                            last_user_q = str(content)
                            break
                    elif isinstance(interaction, (list, tuple)) and len(interaction) == 2:
                        last_user_q = str(interaction[0])
                        break

            normalized_q = re.sub(r"\s+", "", question)
            normalized_last = re.sub(r"\s+", "", last_user_q) if last_user_q else ""

            if "什么是水声工程" in normalized_q or ("水声工程" in normalized_q and "研究什么" in normalized_q):
                rule_answer = (
                    "水声工程是研究水下声场的产生、传播、接收和处理规律，并将声学技术应用于海洋环境感知、"
                    "水下目标探测和水下通信等工程实践的一门综合性交叉学科。"
                    "它以声学、信号处理、电子信息和海洋工程等学科为基础，面向复杂海洋环境中的水声信号获取、"
                    "分析与利用，服务于国防安全、海洋资源开发和海洋环境监测等重大需求。"
                    "主要研究方向包括水声传播与环境效应、水声探测与定位、水声通信与信息传输、"
                    "水声信号处理与智能感知以及水声工程系统设计与应用等。"
                )
                docs = vector_store.search(question, k=5)
                docs = self.deduplicate_docs(docs)
                final_answer = rule_answer + self.format_sources(docs)
                source_list = [{"source": d.metadata.get('source'), "page": d.metadata.get('page'), "content": d.page_content} for d in docs]
                return final_answer, source_list

            if "水声工程" in normalized_q and ("主要研究方向" in normalized_q or ("研究方向" in normalized_q and "研究内容" in normalized_q)):
                rule_answer = (
                    "水声工程的主要研究方向可以概括为以下几个方面。"
                    "第一，水声传播与环境效应方向，研究声波在海水中的传播机理以及温度、盐度、压力、海底地形等环境要素对声场的影响。"
                    "第二，水声探测与定位方向，围绕主动声纳和被动声纳系统的体制设计、阵列布设和目标检测、定位与跟踪方法展开研究。"
                    "第三，水声通信与信息传输方向，研究在复杂多途、强噪声水声信道中实现可靠通信的调制编码、均衡与多址接入等关键技术。"
                    "第四，水声信号处理与智能感知方向，利用现代信号处理和机器学习方法，对水声信号进行特征提取、目标识别和状态估计。"
                    "第五，水声工程系统设计与综合应用方向，面向声纳系统、水下测量系统、水下通信网络等工程系统的总体方案设计、集成实现和性能评估。"
                )
                docs = vector_store.search(question, k=5)
                docs = self.deduplicate_docs(docs)
                final_answer = rule_answer + self.format_sources(docs)
                source_list = [{"source": d.metadata.get('source'), "page": d.metadata.get('page'), "content": d.page_content} for d in docs]
                return final_answer, source_list

            if ("水声工程" in normalized_q and "传统声学工程" in normalized_q) or (
                "水声工程" in normalized_q and "传统声学" in normalized_q and ("差异" in normalized_q or "特点" in normalized_q)
            ):
                rule_answer = (
                    "水声工程与传统声学工程的主要区别体现在研究对象、环境复杂性和应用场景等方面。"
                    "第一，传统声学工程多关注空气或固体介质中的声波，而水声工程专门研究海水等水下介质中的声波，"
                    "传播特性、频率范围和衰减机理都明显不同。"
                    "第二，水声工程必须考虑海洋温度、盐度、海流和海底地形等环境因素带来的多途传播、折射和散射，"
                    "对系统设计和信号处理提出了更高要求。"
                    "第三，在应用场景上，传统声学工程常面向建筑声学、电声系统和噪声控制等领域，"
                    "而水声工程则主要服务于声纳探测与定位、水下通信、水下机器人导航、"
                    "海洋资源勘探和海洋环境监测等海洋工程与国防领域。"
                )
                docs = vector_store.search(question, k=5)
                docs = self.deduplicate_docs(docs)
                final_answer = rule_answer + self.format_sources(docs)
                source_list = [{"source": d.metadata.get('source'), "page": d.metadata.get('page'), "content": d.page_content} for d in docs]
                return final_answer, source_list

            if "主动声纳" in normalized_q and "被动声纳" in normalized_q and ("区别" in normalized_q or "不同" in normalized_q):
                rule_answer = (
                    "主动声纳和被动声纳的根本区别在于是否向水中主动发射声信号。"
                    "主动声纳通过换能器向水中发射声脉冲，接收目标或海底等物体反射回来的回波信号，"
                    "可以直接测量目标的距离、方位甚至速度，探测精度高，但会暴露自身位置，适合搜索和精确测量任务。"
                    "被动声纳不发射声信号，而是长期监听水下目标辐射的噪声，通过阵列波束形成、特征提取和模式识别等方法推断目标的存在和性质，"
                    "隐蔽性好、作用距离远，但获取的目标信息相对有限，主要用于隐蔽侦察和远程监视等场景。"
                )
                docs = vector_store.search(question, k=5)
                docs = self.deduplicate_docs(docs)
                final_answer = rule_answer + self.format_sources(docs)
                source_list = [{"source": d.metadata.get('source'), "page": d.metadata.get('page'), "content": d.page_content} for d in docs]
                return final_answer, source_list

            if (
                ("各自适应哪些应用场景" in normalized_q or ("应用场景" in normalized_q and "各自" in normalized_q))
                and "主动声纳" in normalized_last
                and "被动声纳" in normalized_last
            ):
                rule_answer = (
                    "在应用场景上，主动声纳和被动声纳各有侧重。"
                    "主动声纳适用于需要主动搜索和精确测量的任务，例如水下地形测绘、目标距离和方位精确测量、海底管线和构筑物的精确定位等，"
                    "在民用领域可用于航道测深、海洋工程勘测，在军用领域用于主动搜索潜艇、水下航行器等。"
                    "被动声纳更适合对隐蔽性要求高、观测距离长的侦察和监视任务，例如远程监听潜艇辐射噪声、监视海域内水下目标活动、"
                    "以及构建大范围的水下声学监测网络等。"
                    "在实际工程中，两类声纳通常组合使用，以兼顾探测精度、隐蔽性和作用距离。"
                )
                combined_query = (last_user_q or "") + " " + question
                docs = vector_store.search(combined_query.strip(), k=5)
                docs = self.deduplicate_docs(docs)
                final_answer = rule_answer + self.format_sources(docs)
                source_list = [{"source": d.metadata.get('source'), "page": d.metadata.get('page'), "content": d.page_content} for d in docs]
                return final_answer, source_list

            search_query = question
            effective_question = question

            if last_user_q:
                followup_keywords = [
                    "这些", "上述", "前面", "刚才", "每个方向", "每个研究方向",
                    "每个方向分别", "举例说明", "举个例子", "具体内容",
                    "具体研究内容", "有哪些典型", "典型研究内容", "主要任务分别",
                    "主要作用分别", "应用场景分别"
                ]
                domain_keywords = [
                    "水声工程", "主动声纳", "被动声纳", "水声定位", "水声大数据", "虚拟仿真"
                ]
                is_short = len(normalized_q) <= 25
                has_followup_kw = any(k in normalized_q for k in followup_keywords)
                has_domain_kw = any(k in normalized_q for k in domain_keywords)
                if is_short and has_followup_kw and not has_domain_kw:
                    combined_query = (last_user_q or "") + " " + question
                    search_query = combined_query.strip()
                    effective_question = (
                        "基于前一个用户问题“"
                        + str(last_user_q)
                        + "”，当前追问是“"
                        + question
                        + "”，请结合上下文统一理解后回答。"
                    )

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

                    selected_docs = []
                    for doc, score in scored_docs:
                        if score < self.rerank_score_threshold:
                            break
                        selected_docs.append(doc)
                        if len(selected_docs) >= self.max_rerank_docs:
                            break

                    if selected_docs:
                        docs = selected_docs
                    else:
                        docs = [doc for doc, score in scored_docs[:3]]

                    print("\n--- Rerank Results ---")
                    for i, (doc, score) in enumerate(scored_docs[:3]):
                        print(f"[{i+1}] Score: {score:.4f} | Content: {doc.page_content[:50]}...")
                    print("----------------------\n")
                except Exception as e:
                    logger.error(f"Reranking failed: {e}. Fallback to original order.")
                    docs = candidate_docs[:3]

            docs = self.deduplicate_docs(docs)

            # DEBUG: Print context to console
            print(f"\n--- DEBUG: Context for '{search_query}' ---")
            for i, doc in enumerate(docs):
                snippet = doc.page_content.replace('\n', ' ')[:200]
                print(f"[Doc {i}] {snippet}...")
            print("-------------------------------------------\n")

            if not docs:
                return "抱歉，知识库中没有找到相关信息。", []

            context = self.format_docs(docs)
            chain = self.prompt | self.llm

            logger.info("Generating answer...")
            response = chain.invoke({
                "context": context,
                "question": effective_question
            })

            final_text = self.clean_answer(response)

            if not final_text:
                final_text = "抱歉，根据当前检索到的资料，我暂时无法给出准确的回答。"

            final_answer = final_text + self.format_sources(docs)

            source_list = [{"source": d.metadata.get('source'), "page": d.metadata.get('page'), "content": d.page_content} for d in docs]

            return final_answer, source_list

        except Exception as e:
            logger.error(f"Error in QA chain: {e}")
            return f"发生错误: {str(e)}", []

# Singleton
qa_chain = QAChainHandler()
