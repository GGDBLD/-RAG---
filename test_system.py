
import time
import sys
import os

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

print("🔄 正在初始化系统组件...")
try:
    from src.qa_chain import qa_chain
    print("✅ 问答链加载成功。")
except Exception as e:
    print(f"❌ 加载问答链失败: {e}")
    sys.exit(1)

def test_question(question, history=None, expected_keywords=None, is_rule_based=False):
    print(f"\n🔹 测试问题: '{question}'")
    if history:
        print(f"   上下文: 包含 {len(history)} 轮历史对话")
    
    start_time = time.time()
    try:
        # Mocking vector store search if needed? No, use real one.
        answer, sources = qa_chain.answer_question(question, chat_history=history)
        duration = time.time() - start_time
        
        print(f"   ⏱️ 耗时: {duration:.2f}秒")
        print(f"   📝 回答长度: {len(answer)} 字符")
        print(f"   📄 参考文档: {len(sources)} 篇")
        
        # Validation
        passed = True
        if not answer or "发生错误" in answer:
            print(f"   ❌ 回答包含错误: {answer}")
            passed = False
        
        if expected_keywords:
            missing = [k for k in expected_keywords if k not in answer]
            if missing:
                print(f"   ⚠️ 警告: 缺少关键词 {missing}")
        
        if is_rule_based and duration > 2.0:
             print("   ⚠️ 警告: 规则回答耗时过长")

        if passed:
            print("   ✅ 测试通过")
        
        return answer, passed, duration

    except Exception as e:
        print(f"   ❌ 异常: {e}")
        return None, False, 0

def run_tests():
    print("🚀 开始自动化系统测试...")
    
    total_tests = 0
    passed_tests = 0
    total_duration = 0
    
    # Test 1: Rule-based (Should be fast)
    ans1, p1, d1 = test_question(
        "什么是声纳方程", 
        expected_keywords=["声纳系统", "声源级"],
        is_rule_based=True
    )
    total_tests += 1
    if p1: passed_tests += 1
    total_duration += d1
    
    # Test 2: Retrieval (General knowledge)
    ans2, p2, d2 = test_question(
        "师国峰的性别",
        expected_keywords=["男"]
    )
    total_tests += 1
    if p2: passed_tests += 1
    total_duration += d2

    # Test 3: Multi-turn (History handling)
    # Using the answer from Test 2 as history
    history = [("师国峰的性别", ans2)]
    ans3, p3, d3 = test_question(
        "他是哪个学校的？",
        history=history,
        expected_keywords=["哈尔滨工程大学"]
    )
    total_tests += 1
    if p3: passed_tests += 1
    total_duration += d3
    
    # Test 4: Garbage Input
    ans4, p4, d4 = test_question(
        "dsjfklsdjflksdjfklsdj"
    )
    total_tests += 1
    if p4: passed_tests += 1
    total_duration += d4

    print("\n" + "="*30)
    print(f"📊 测试总结")
    print(f"总测试数: {total_tests}")
    print(f"通过: {passed_tests}")
    print(f"总耗时: {total_duration:.2f}秒")
    print("="*30)

if __name__ == "__main__":
    run_tests()
    input("\n按回车键退出...")
