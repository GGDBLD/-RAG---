import ollama
from src.vector_store import search_similar_text

def build_prompt(query: str, similar_docs: list) -> str:
    """
    构建提示词：结合检索结果和问题
    """
    # 拼接上下文
    context = "\n\n".join([f"【来源：{doc['source']}】{doc['text']}" for doc in similar_docs])
    
    # 提示词模板（适配Qwen模型）
    prompt = f"""
你是一个水声工程领域的专业助手，仅基于提供的上下文回答问题，不要编造信息。
如果上下文没有相关信息，直接回答“未找到相关答案”。

### 上下文：
{context}

### 问题：
{query}

### 回答：
"""
    return prompt.strip()

def get_answer(query: str) -> str:
    """
    核心问答逻辑：检索+生成回答
    """
    # 1. 检索相似文本
    similar_docs = search_similar_text(query)
    if not similar_docs:
        return "未找到相关答案"
    
    # 2. 构建提示词
    prompt = build_prompt(query, similar_docs)
    
    # 3. 调用本地Qwen模型生成回答
    try:
        response = ollama.chat(
            model='qwen:1.8b-chat',
            messages=[{'role': 'user', 'content': prompt}]
        )
        answer = response['message']['content']
        
        # 添加来源标注
        sources = list(set([doc['source'] for doc in similar_docs]))
        answer += f"\n\n【信息来源】：{', '.join(sources)}"
        
        return answer
    except Exception as e:
        print(f"调用模型失败：{e}")
        return f"回答生成失败：{str(e)}"

if __name__ == "__main__":
    # 测试问答
    test_query = "海水声速的计算公式是什么？"
    answer = get_answer(test_query)
    print(f"问题：{test_query}")
    print(f"回答：{answer}")