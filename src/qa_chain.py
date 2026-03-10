from typing import Tuple, List, Dict, Generator
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from src.vector_store import vector_store
from src.utils import setup_logger
import os
import re
from sentence_transformers import CrossEncoder
from functools import lru_cache
from src.acoustic_tools import AcousticCalculator

logger = setup_logger('qa_chain')

class QAChainHandler:
    def __init__(self):
        self.reranker_path = r"e:\rag_project\models\bge-reranker-base"
        self.reranker = None
        self.rerank_score_threshold = 0.0
        self.max_rerank_docs = 3 # Reduce to 3 for faster inference
        if os.path.exists(self.reranker_path):
            try:
                logger.info(f"Loading Reranker model from {self.reranker_path}...")
                self.reranker = CrossEncoder(self.reranker_path)
                logger.info("Reranker loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load Reranker: {e}")
        else:
            logger.warning(f"Reranker model not found at {self.reranker_path}. Running in retrieval-only mode.")

        # 配置 Qwen-Plus (通义千问)
        self.llm = ChatOpenAI(
            api_key="sk-3034bf5ffdb84bb291bff7f41cd1d302", # 已直接配置 API KEY
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model="qwen-plus",
            temperature=0.2,
            max_tokens=2048
        )

        template = """你是水声工程领域的中文问答助手。
【当前应用场景】：
- 应用环境：{env_context}
- 设备类型：{device_context}

请结合上述【应用场景】和下方的【已知信息】来回答【问题】。

回答要求：
1. 优先使用【已知信息】中的内容。
2. 必须针对【当前应用场景】进行回答。例如：如果是“浅海探测”，请重点考虑多途效应和浅海声传播特性；如果是“被动声纳”，请重点关注辐射噪声和检测阈。
3. 如果【已知信息】中没有关于该场景的具体描述，请基于你的水声专业知识进行合理的推断，但要明确告知用户这是推断。
4. 如果【已知信息】完全不相关，请回答：“根据当前知识库暂时无法回答该问题。”
5. 回答要简洁直接：先给结论，再给必要步骤或要点；不要写长篇背景综述。
6. 禁止写“根据……得知/结合片段/如上所述/由文献可知”等冗余套话；不要引用“片段编号”或复述【已知信息】原文。
7. 默认输出不超过 10 行，除非用户明确要求展开。
8. 如果【问题】属于“公式/数值计算类问题”（例如包含：计算、求、估算、单位、dB、Hz、kHz、m/s、km、log、Δf、TL、SNR 等），请进入【计算模式】输出，避免长篇解释。

【计算模式】输出格式（严格遵守）：
【结果】
只给最终数值结果+单位；如给范围则写范围+单位。

【已知量】
逐条列出已知参数与单位；若题目缺少关键参数（如角度、入射方向、是否同向/相向），必须明确指出缺失项。

【核心公式】
写出 1～2 个最关键公式（可用 LaTeX 形式）。

【代入计算】
给出代入后的算式，并逐步计算到数值结果（保留必要小数位）。
若存在不确定参数，请给出可计算的“最大/最小/范围”，并写清假设（例如取 cosθ=1 表示正对运动）。

【已知信息】：
{context}

【问题】：
{question}

【回答】："""

        self.prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question", "env_context", "device_context"]
        )

        # Pre-compile regex for performance
        self.entity_pattern = re.compile(r'[a-zA-Z0-9]{2,}')
        self.split_pattern = re.compile(r'(?<=[。！？!?])')

    def format_docs(self, docs: List[Document]) -> str:
        return "\n\n".join(doc.page_content.replace('\n', ' ') for doc in docs)

    def format_sources(self, docs: List[Document]) -> str:
        if not docs:
            return ""

        # Optimize source formatting with set comprehension
        sources = {f"{doc.metadata.get('source', 'Unknown')} (Page {doc.metadata.get('page', 'N/A')})" for doc in docs}
        return "\n\n(信息来源: " + ", ".join(sources) + ")"

    def clean_answer(self, text: str, max_chars: int = 2000) -> str:
        stripped = text.strip()
        if not stripped:
            return stripped
        # Remove repetitive cleaning logic that might break markdown or math formulas
        return stripped[:max_chars]

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

    @lru_cache(maxsize=100)
    def _cached_rerank(self, query: str, doc_contents: tuple) -> list:
        """Cache reranking results for identical query-document pairs"""
        if not self.reranker:
            return []
        # Reconstruct pairs for prediction
        pairs = [[query, content] for content in doc_contents]
        return self.reranker.predict(pairs)

    def _get_retrieval_context(self, question: str, chat_history: List[Tuple[str, str]] = None) -> Tuple[List[Document], str, str]:
        """Helper to retrieve documents and check rules"""
        logger.info(f"Processing question: {question}")
        
        # 1. Check for rule-based answers
        last_user_q = None
        if chat_history:
            # Optimized history extraction
            for interaction in reversed(chat_history[-3:]):
                if isinstance(interaction, (list, tuple)) and len(interaction) == 2:
                    last_user_q = str(interaction[0])
                    break
                elif isinstance(interaction, dict) and interaction.get("role") == "user":
                    last_user_q = str(interaction.get("content", ""))
                    break

        normalized_q = re.sub(r"\s+", "", question)
        
        # --- Restore Rule Based Logic ---
        rule_answer = None
        
        if "声纳方程" in normalized_q or "声呐方程" in normalized_q:
            rule_answer = (
                "声纳方程是用来描述声纳系统中各个关键声学量之间关系的工程公式，以能量平均意义上给出声纳能够实现探测或通信的条件。"
                "在主动声纳中，声纳方程通常把声源级、传播损失、目标强度、混响或噪声级、指向性指数和检测门限联系起来，"
                "用于估算在给定声场和设备条件下的可探测距离或所需声源级。"
                "在被动声纳中，声纳方程则将目标辐射噪声级、环境背景噪声级、阵列增益和处理增益等量联系起来，"
                "用于分析在某一信噪比要求下被动侦听的作用距离和探测概率。"
                "经典声纳方程形式简洁、物理意义清晰，是声纳系统设计、性能评估和战术使用分析的基础工具。"
            )
        elif "什么是水声工程" in normalized_q or ("水声工程" in normalized_q and "研究什么" in normalized_q):
            rule_answer = (
                "水声工程是研究水下声场的产生、传播、接收和处理规律，并将声学技术应用于海洋环境感知、"
                "水下目标探测和水下通信等工程实践的一门综合性交叉学科。"
                "它以声学、信号处理、电子信息和海洋工程等学科为基础，面向复杂海洋环境中的水声信号获取、"
                "分析与利用，服务于国防安全、海洋资源开发和海洋环境监测等重大需求。"
                "主要研究方向包括水声传播与环境效应、水声探测与定位、水声通信与信息传输、"
                "水声信号处理与智能感知以及水声工程系统设计与应用等。"
            )
        elif "水声工程" in normalized_q and ("主要研究方向" in normalized_q or ("研究方向" in normalized_q and "研究内容" in normalized_q)):
            rule_answer = (
                "水声工程的主要研究方向可以概括为以下几个方面。"
                "第一，水声传播与环境效应方向，研究声波在海水中的传播机理以及温度、盐度、压力、海底地形等环境要素对声场的影响。"
                "第二，水声探测与定位方向，围绕主动声纳和被动声纳系统的体制设计、阵列布设和目标检测、定位与跟踪方法展开研究。"
                "第三，水声通信与信息传输方向，研究在复杂多途、强噪声水声信道中实现可靠通信的调制编码、均衡与多址接入等关键技术。"
                "第四，水声信号处理与智能感知方向，利用现代信号处理和机器学习方法，对水声信号进行特征提取、目标识别和状态估计。"
                "第五，水声工程系统设计与综合应用方向，面向声纳系统、水下测量系统、水下通信网络等工程系统的总体方案设计、集成实现和性能评估。"
            )
        elif ("水声工程" in normalized_q and "传统声学工程" in normalized_q) or (
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
            
        if rule_answer:
            docs = vector_store.search(question, k=5)
            docs = self.deduplicate_docs(docs)
            return docs, rule_answer, question

        effective_question = question
        # Optimize history handling - only use last turn if history is long
        search_query = question
        if chat_history:
            recent_history = chat_history[-2:] # Only look at last 2 turns
            history_context_parts = []
            for item in recent_history:
                # Robust unpacking for various history formats
                q, a = "", ""
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    q, a = str(item[0]), str(item[1])
                elif isinstance(item, dict):
                    # Handle potential dict format if history structure changes
                    q = str(item.get("question", item.get("human", "")))
                    a = str(item.get("answer", item.get("ai", "")))
                
                if q and a:
                    history_context_parts.append(f"Human: {q}\nAI: {a}")
            
            if history_context_parts:
                history_context = "\n".join(history_context_parts)
                # Simple concatenation is faster than LLM condensation
                search_query = f"{history_context}\nHuman: {question}"[-512:] # Truncate to avoid token limits
        
        # 2. Query Boost from scene tags
        boost_terms = []
        m_env = re.search(r"\[当前场景：(.*?)\]", question)
        m_dev = re.search(r"\[设备类型：(.*?)\]", question)
        m_ss = re.search(r"\[海况：(.*?)\]", question)
        m_bt = re.search(r"\[海底：(.*?)\]", question)
        m_sp = re.search(r"\[声速剖面：(.*?)\]", question)
        m_fb = re.search(r"\[频段：(.*?)\]", question)
        m_task = re.search(r"\[任务：(.*?)\]", question)
        if m_env:
            v = m_env.group(1)
            if "浅海" in v:
                boost_terms += ["浅海", "多途", "混响", "海底反射"]
            if "深海" in v or "汇聚区" in v:
                boost_terms += ["深海", "汇聚区", "声道", "SOFAR"]
            if "冰下" in v:
                boost_terms += ["冰下", "界面散射"]
        if m_dev:
            v = m_dev.group(1)
            if "被动" in v:
                boost_terms += ["被动", "辐射噪声", "阵列增益", "检测阈"]
            if "主动" in v:
                boost_terms += ["主动", "目标强度", "混响", "回波"]
        if m_ss:
            boost_terms += ["海况", str(m_ss.group(1))]
        if m_bt:
            v = m_bt.group(1)
            if v in ["泥", "砂", "岩"]:
                boost_terms += [v + "底", "底反射", "底散射"]
        if m_sp:
            v = m_sp.group(1)
            if "表面声道" in v:
                boost_terms += ["表面声道", "声速剖面"]
            if "中层极小" in v:
                boost_terms += ["声速极小", "折射", "声速剖面"]
            if "汇聚区" in v:
                boost_terms += ["汇聚区", "声道", "SOFAR"]
        if m_fb:
            v = m_fb.group(1)
            if v in ["低频", "中频", "高频"]:
                boost_terms += [v, "吸收", "指向性"]
        if m_task:
            v = m_task.group(1)
            boost_terms += [v]
        if boost_terms:
            prefix = " ".join(boost_terms[:12])
            search_query = (prefix + " " + search_query)[-768:]

        # Reduce initial retrieval count to speed up reranking
        initial_k = 10 # Reduced from default/larger value
        candidate_docs = vector_store.search(search_query, k=initial_k)
        
        docs = candidate_docs
        if self.reranker and candidate_docs:
            logger.info("Reranking documents...")
            try:
                # Use cached reranking
                doc_contents = tuple(doc.page_content for doc in candidate_docs)
                scores = self._cached_rerank(search_query, doc_contents)
                
                scored_docs = []
                for i, doc in enumerate(candidate_docs):
                    s = scores[i] if i < len(scores) else 0.0
                    bonus = 0.0
                    if boost_terms:
                        if any(term for term in boost_terms if term and term in doc.page_content):
                            bonus = 0.05 if s >= 0 and s <= 1.0 else 0.5
                        # Additional bonus if metadata matches scene tags
                        try:
                            meta = getattr(doc, "metadata", {}) or {}
                            meta_vals = [str(v) for k, v in meta.items() if k in ["env","device","band","ssp_type","bottom_type","task","array_type"] and v]
                            if meta_vals and any(mv in boost_terms for mv in meta_vals):
                                bonus += (0.05 if s >= 0 and s <= 1.0 else 0.5)
                        except Exception:
                            pass
                    scored_docs.append((doc, s + bonus))
                scored_docs.sort(key=lambda x: x[1], reverse=True)

                selected_docs = []
                if scored_docs:
                    top_score = scored_docs[0][1]
                    
                for doc, score in scored_docs:
                    # 1. Absolute threshold filtering
                    if score < self.rerank_score_threshold:
                        break
                        
                    # 2. Relative threshold filtering (Adaptive)
                    # Updated to handle both logits (diff > 2.5) and probabilities (diff > 0.5)
                    # If score is > 0.9, it's likely a probability.
                    diff = top_score - score
                    is_prob = top_score > 0.9 and score >= 0 and score <= 1.0
                    
                    threshold = 0.5 if is_prob else 2.5
                    
                    if diff > threshold:
                        logger.info(f"Skipping doc with score {score:.4f} (too far from top {top_score:.4f}, threshold {threshold})")
                        break
                    
                    # 3. Keyword mismatch check using pre-compiled regex
                    q_entities = self.entity_pattern.findall(search_query) 
                    if q_entities:
                        has_any_entity = any(e in doc.page_content for e in q_entities)
                        if not has_any_entity:
                            # Stricter relative threshold for non-entity docs
                            strict_threshold = 0.3 if is_prob else 1.5
                            if diff > strict_threshold:
                                logger.info(f"Skipping doc {score:.4f} (missing entities {q_entities} and score too low)")
                                continue
                    
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

        return docs, rule_answer, effective_question

    def answer_question(self, question: str, chat_history: List[Tuple[str, str]] = None) -> Tuple[str, List[Dict]]:
        try:
            docs, rule_answer, effective_question = self._get_retrieval_context(question, chat_history)

            # Rule-based: DI收益（避免臆造比例）
            if not rule_answer:
                if "指向性指数" in effective_question or "DI" in effective_question:
                    rule_answer = (
                        "结论：DI 每增加 x dB，等效 SNR 增加 x dB；作用距离提升幅度取决于传播损失与噪声模型，"
                        "需具体参数方可量化。建议先给出 SL、TL、NL、DI、DT，再按声纳方程评估。"
                    )

            if rule_answer:
                final_answer = rule_answer + self.format_sources(docs)
                source_list = [{"source": d.metadata.get('source'), "page": d.metadata.get('page'), "content": d.page_content} for d in docs]
                return final_answer, source_list

            if not docs:
                return "抱歉，知识库中没有找到相关信息。", []

            # Calculation intent router: prefer deterministic calculator output
            calc_text = self._try_calculation_answer(effective_question)
            if calc_text:
                final_answer = calc_text + self.format_sources(docs)
                source_list = [{"source": d.metadata.get('source'), "page": d.metadata.get('page'), "content": d.page_content} for d in docs]
                return final_answer, source_list

            context = self.format_docs(docs)
            chain = self.prompt | self.llm

            logger.info("Generating answer...")
            env_context = "通用/默认"
            device_context = "未知"
            m_env = re.search(r"\[当前场景：(.*?)\]", effective_question)
            if m_env:
                env_context = m_env.group(1)
                effective_question = effective_question.replace(m_env.group(0), "").strip()
            m_dev = re.search(r"\[设备类型：(.*?)\]", effective_question)
            if m_dev:
                device_context = m_dev.group(1)
                effective_question = effective_question.replace(m_dev.group(0), "").strip()
            response = chain.invoke({"context": context, "question": effective_question, "env_context": env_context, "device_context": device_context})
            
            # Handle AIMessage object from ChatOpenAI
            if hasattr(response, 'content'):
                response = response.content
            else:
                response = str(response)

            final_text = self.clean_answer(response, max_chars=3000)

            if not final_text:
                final_text = "抱歉，根据当前检索到的资料，我暂时无法给出准确的回答。"

            final_answer = final_text + self.format_sources(docs)

            source_list = [{"source": d.metadata.get('source'), "page": d.metadata.get('page'), "content": d.page_content} for d in docs]

            return final_answer, source_list

        except Exception as e:
            logger.error(f"Error in QA chain: {e}")
            return f"发生错误: {str(e)}", []

    def answer_question_stream(self, question: str, chat_history: List[Tuple[str, str]] = None) -> Generator[Tuple[str, List[Dict]], None, None]:
        try:
            # 解析 Context Injection (从 question 中提取场景信息)
            env_context = "通用/默认"
            device_context = "未知"
            effective_question = question

            # 简单的正则提取 (对应 app.py 中的 context_prefix)
            match_env = re.search(r"\[当前场景：(.*?)\]", question)
            if match_env:
                env_context = match_env.group(1)
                effective_question = effective_question.replace(match_env.group(0), "").strip()
            
            match_dev = re.search(r"\[设备类型：(.*?)\]", question)
            if match_dev:
                device_context = match_dev.group(1)
                effective_question = effective_question.replace(match_dev.group(0), "").strip()

            # 计算类与部分高频结论：优先走确定性路径，保证与计算器一致
            if "指向性指数" in effective_question or "DI" in effective_question:
                yield (
                    "结论：DI 每增加 x dB，等效 SNR 增加 x dB；作用距离提升幅度取决于传播损失与噪声模型，"
                    "需具体参数方可量化。建议给出 SL、TL、NL、DI、DT，再按声纳方程评估。",
                    []
                )
                return

            calc_text = self._try_calculation_answer(effective_question)
            if calc_text:
                yield calc_text, []
                return

            docs, rule_answer, _ = self._get_retrieval_context(question, chat_history)

            if rule_answer:
                final_answer = rule_answer + self.format_sources(docs)
                source_list = [{"source": d.metadata.get('source'), "page": d.metadata.get('page'), "content": d.page_content} for d in docs]
                yield final_answer, source_list
                return

            if not docs:
                yield "抱歉，知识库中没有找到相关信息。", []
                return

            context = self.format_docs(docs)
            chain = self.prompt | self.llm

            logger.info(f"Generating answer stream... (Env: {env_context}, Device: {device_context})")
            
            full_response = ""
            for chunk in chain.stream({
                "context": context,
                "question": effective_question,
                "env_context": env_context,
                "device_context": device_context
            }):
                # Handle both string (Ollama) and AIMessageChunk (ChatOpenAI)
                content = chunk.content if hasattr(chunk, 'content') else str(chunk)
                full_response += content
                yield full_response, []

            final_text = self.clean_answer(full_response, max_chars=3000)

            if not final_text:
                final_text = "抱歉，根据当前检索到的资料，我暂时无法给出准确的回答。"

            final_answer = final_text + self.format_sources(docs)

            source_list = [{"source": d.metadata.get('source'), "page": d.metadata.get('page'), "content": d.page_content} for d in docs]

            yield final_answer, source_list

        except Exception as e:
            logger.error(f"Error in QA chain stream: {e}")
            yield f"发生错误: {str(e)}", []

    def _try_calculation_answer(self, q: str) -> str:
        """Parse calculation intent from question and return concise result using AcousticCalculator"""
        try:
            text = q
            # Transmission Loss
            if ("计算传播损失" in text) or (("传播损失" in text or "TL" in text) and ("距离" in text)):
                import re
                r_km = None; f_khz = 0.0; tl_type = "spherical"
                # distance: km or m
                m_r_km = re.search(r"距离\s*([0-9]+(?:\.[0-9]+)?)\s*km", text)
                m_r_m = re.search(r"距离\s*([0-9]+(?:\.[0-9]+)?)\s*m(?![a-zA-Z])", text)
                if m_r_km:
                    r_km = float(m_r_km.group(1))
                elif m_r_m:
                    r_km = float(m_r_m.group(1)) / 1000.0
                # frequency: kHz or Hz
                m_f_khz = re.search(r"频率\s*([0-9]+(?:\.[0-9]+)?)\s*kHz", text)
                m_f_hz = re.search(r"频率\s*([0-9]+(?:\.[0-9]+)?)\s*Hz", text)
                if m_f_khz:
                    f_khz = float(m_f_khz.group(1))
                elif m_f_hz:
                    f_khz = float(m_f_hz.group(1)) / 1000.0
                if "球面扩展" in text: tl_type = "spherical"
                elif "柱面扩展" in text: tl_type = "cylindrical"
                elif "混合扩展" in text or "hybrid" in text: tl_type = "hybrid"
                if r_km:
                    return AcousticCalculator.calc_transmission_loss(r_km, f_khz, tl_type)
                if "计算传播损失" in text:
                    return "需要参数：距离 (km)，可选频率 (kHz) 与扩展类型（球面/柱面/混合）。"
                return None
            # Sonar Equation
            if ("声纳方程" in text and "计算" in text) or ("SNR" in text and "计算" in text):
                import re
                is_active = "主动" in text
                def pick(pattern):
                    m = re.search(pattern, text)
                    return float(m.group(1)) if m else None
                sl = pick(r"SL\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*dB")
                tl = pick(r"TL\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*dB")
                nl = pick(r"NL\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*dB")
                di = pick(r"DI\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*dB")
                ts = pick(r"TS\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*dB") or 0.0
                if sl is not None and tl is not None and nl is not None and di is not None:
                    return AcousticCalculator.calc_sonar_equation(sl, tl, nl, di, ts, is_active)
                if "计算" in text:
                    return "需要参数：SL、TL、NL、DI（dB），主动模式下可选 TS（dB）。"
                return None
            # Doppler
            if "多普勒" in text:
                import re
                vs = None; vt = None; f0 = None
                m_vs_kn = re.search(r"声源速度\s*([0-9]+(?:\.[0-9]+)?)\s*节", text)
                m_vt_kn = re.search(r"目标速度\s*([0-9]+(?:\.[0-9]+)?)\s*节", text)
                m_vs_ms = re.search(r"声源速度\s*([0-9]+(?:\.[0-9]+)?)\s*m/s", text)
                m_vt_ms = re.search(r"目标速度\s*([0-9]+(?:\.[0-9]+)?)\s*m/s", text)
                m_f0 = re.search(r"中心频率\s*([0-9]+(?:\.[0-9]+)?)\s*Hz", text)
                KNOT_PER_MS = 1.0 / 0.51444
                if m_vs_kn: vs = float(m_vs_kn.group(1))
                elif m_vs_ms: vs = float(m_vs_ms.group(1)) * KNOT_PER_MS
                if m_vt_kn: vt = float(m_vt_kn.group(1))
                elif m_vt_ms: vt = float(m_vt_ms.group(1)) * KNOT_PER_MS
                if m_f0: f0 = float(m_f0.group(1))
                if vs is not None and vt is not None and f0 is not None:
                    return AcousticCalculator.calc_doppler_shift(vs, vt, f0)
                if "计算" in text:
                    return "需要参数：声源速度 (节)、目标速度 (节)、中心频率 (Hz)。"
                return None
            # Inverse Max Range
            if "最大探测距离" in text or "逆向求解" in text:
                import re
                fom = None; f_khz = 1.0; t = "spherical"
                m_fom = re.search(r"FOM\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*dB", text)
                m_fk = re.search(r"频率\s*([0-9]+(?:\.[0-9]+)?)\s*kHz", text)
                m_fh = re.search(r"频率\s*([0-9]+(?:\.[0-9]+)?)\s*Hz", text)
                if m_fom: fom = float(m_fom.group(1))
                if m_fk: f_khz = float(m_fk.group(1))
                elif m_fh: f_khz = float(m_fh.group(1)) / 1000.0
                if "柱面扩展" in text: t = "cylindrical"
                elif "混合扩展" in text or "hybrid" in text: t = "hybrid"
                if fom is not None:
                    return AcousticCalculator.solve_max_range(fom, f_khz, t)
                if "逆向求解" in text or "最大探测距离" in text:
                    return "需要参数：FOM（dB），可选频率 (kHz) 与扩展类型。"
                return None
        except Exception:
            return None
        return None
# Singleton
qa_chain = QAChainHandler()
