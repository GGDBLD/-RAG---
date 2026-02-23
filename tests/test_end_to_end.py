import sys
import os

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
        import time

        def run_case(label, question, history):
            print(f"\n3. [Case {label}] 提问: '{question}'", flush=True)
            start_time = time.time()
            answer, sources = qa.answer_question(question, history)
            end_time = time.time()
            duration = end_time - start_time
            print(f"   耗时: {duration:.2f}s", flush=True)
            print(f"   回答: {answer}", flush=True)
            print(f"   来源: {[s.get('source') for s in sources]}", flush=True)
            if "暂时无法回答" in answer or "没有找到相关信息" in answer or not answer.strip():
                print("   结果: ❌ 可能未从知识库中得到有效回答", flush=True)
            else:
                print("   结果: ✅ 已生成有效回答", flush=True)
            history.append((question, answer))
            return history

        history = []

        history = run_case("1-单轮-舰船噪声因素", "在复杂海洋环境下，影响舰船水下噪声传播的主要因素有哪些？", history)
        history = run_case("2-单轮-水声定位油气开采", "水声定位系统在海洋油气开采作业中通常如何使用？", history)
        history = run_case("3-单轮-大数据平台架构", "简要介绍水声大数据平台的总体功能架构。", history)

        history = run_case("4-多轮-大数据平台追问", "其中哪些功能与航行安全保障直接相关？", history)

        history = run_case("5-单轮-虚拟实验教学", "被动声呐虚拟仿真实验的教学目标和主要内容是什么？", history)
        history = run_case("6-单轮-无关问题控制", "水声工程中火箭发动机设计的关键问题有哪些？", history)

    except Exception as e:
        print(f"\n❌ 测试过程中发生崩溃: {e}", flush=True)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_end_to_end_test()
