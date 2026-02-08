import sys
import os

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_end_to_end_test():
    print("=== RAG 系统全链路测试 (End-to-End) ===", flush=True)
    
    try:
        print("1. [Import] 正在加载 QA 模块...", flush=True)
        from src.qa_chain import QAChainHandler
        print("   ✅ 加载成功。", flush=True)
        
        print("2. [Init] 正在初始化 QA 链 (加载模型)...", flush=True)
        qa = QAChainHandler()
        print("   ✅ 初始化成功。", flush=True)
        
        # 测试问题
        question = "什么是声纳方程？"
        print(f"\n3. [Test] 正在提问: '{question}'", flush=True)
        
        # 计时
        import time
        start_time = time.time()
        
        # 调用问答
        # 注意：qa_chain 内部会调用 vector_store 进行检索
        answer, sources = qa.answer_question(question, [])
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n=== 测试结果 (耗时: {duration:.2f}s) ===", flush=True)
        print(f"【问题】: {question}", flush=True)
        print(f"【回答】: {answer}", flush=True)
        print(f"【来源】: {[s.get('source') for s in sources]}", flush=True)
        
        # 验证逻辑
        if "无法找到" in answer or not answer.strip():
            print("\n❌ 测试失败: 回答无效或未找到答案。", flush=True)
        else:
            print("\n✅ 测试通过: 系统成功生成了回答。", flush=True)

    except Exception as e:
        print(f"\n❌ 测试过程中发生崩溃: {e}", flush=True)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_end_to_end_test()