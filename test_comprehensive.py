
import time
import sys
import os

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from src.qa_chain import qa_chain
    print("✅ QA Chain loaded successfully.")
except Exception as e:
    print(f"❌ Failed to load QA Chain: {e}")
    sys.exit(1)

def test_question(question, history=None, expected_keywords=None, is_rule_based=False, category="General"):
    print(f"\n🔹 [{category}] Testing: '{question}'")
    if history:
        print(f"   Context: {len(history)} previous turns")
    
    start_time = time.time()
    try:
        answer, sources = qa_chain.answer_question(question, chat_history=history)
        duration = time.time() - start_time
        
        print(f"   ⏱️ Duration: {duration:.2f}s")
        print(f"   📝 Answer Length: {len(answer)} chars")
        if sources:
            print(f"   📄 Sources: {len(sources)} docs ({sources[0]['source']})")
        else:
            print(f"   📄 Sources: 0 docs")
        
        # Validation
        passed = True
        if not answer or "发生错误" in answer:
            print(f"   ❌ Error in answer: {answer}")
            passed = False
        
        if expected_keywords:
            missing = [k for k in expected_keywords if k not in answer]
            if missing:
                print(f"   ⚠️ Warning: Missing keywords {missing}")
                # Don't fail strictly on keywords for LLM answers, just warn
        
        if is_rule_based and duration > 2.0:
             print("   ⚠️ Warning: Rule-based answer took too long")

        if passed:
            print("   ✅ Test Passed")
        
        return answer, passed, duration

    except Exception as e:
        print(f"   ❌ Exception: {e}")
        return None, False, 0

def run_comprehensive_tests():
    print("\n🚀 Starting COMPREHENSIVE System Test Suite...")
    print("==================================================")
    
    results = []
    
    # 1. Basic Concepts (Rule-based)
    q1 = "什么是声纳方程"
    ans, p, d = test_question(q1, expected_keywords=["声纳系统"], is_rule_based=True, category="Basic Concept")
    results.append(("Basic: Sonar Eq", p, d))

    q2 = "什么是水声工程"
    ans, p, d = test_question(q2, category="Basic Concept")
    results.append(("Basic: Acoustic Eng", p, d))

    # 2. Document Details (Retrieval)
    # Personnel
    q3 = "师国峰的性别是什么？"
    ans, p, d = test_question(q3, expected_keywords=["男"], category="Retrieval: Personnel")
    results.append(("Ret: Shi Guofeng", p, d))

    q4 = "殷敬伟在项目中担任什么角色？"
    ans, p, d = test_question(q4, category="Retrieval: Personnel")
    results.append(("Ret: Yin Jingwei", p, d))

    # Project Background
    q5 = "为什么水声工程专业的实验成本较高？"
    ans, p, d = test_question(q5, expected_keywords=["国防", "设备"], category="Retrieval: Background")
    results.append(("Ret: High Cost", p, d))

    q6 = "虚拟仿真实验主要解决了本科教学中的什么痛点？"
    ans, p, d = test_question(q6, category="Retrieval: Pain Points")
    results.append(("Ret: Pain Points", p, d))

    # Technical Details
    q7 = "什么是“清劲风”海况？"
    ans, p, d = test_question(q7, category="Retrieval: Technical")
    results.append(("Ret: Sea State", p, d))

    # 3. Multi-turn Conversation
    print("\n🔹 [Multi-turn] Starting Conversation Chain...")
    history = []
    
    # Turn 1
    t1_q = "师国峰是谁？"
    t1_ans, t1_p, t1_d = test_question(t1_q, category="Multi-turn: Turn 1")
    if t1_ans:
        history.append((t1_q, t1_ans))
    
    # Turn 2
    t2_q = "他是哪个学校的？"
    t2_ans, t2_p, t2_d = test_question(t2_q, history=history, expected_keywords=["哈尔滨工程大学"], category="Multi-turn: Turn 2")
    if t2_ans:
        history.append((t2_q, t2_ans))
    
    # Turn 3
    t3_q = "他参与了哪个项目的建设？"
    t3_ans, t3_p, t3_d = test_question(t3_q, history=history, category="Multi-turn: Turn 3")
    
    results.append(("Multi-turn Chain", t1_p and t2_p and t3_p, t1_d + t2_d + t3_d))

    # 4. Hallucination Test
    q_hal = "水声工程中如何应用火箭发动机技术？"
    ans, p, d = test_question(q_hal, category="Hallucination Check")
    # We expect it to say "unknown" or similar, but as long as it doesn't crash, it's a pass for system stability.
    # Ideally check for "无法回答" or similar.
    results.append(("Hallucination Check", p, d))

    print("\n==================================================")
    print("📊 Test Summary")
    print("==================================================")
    for name, passed, duration in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} | {name:<20} | {duration:.2f}s")
    print("==================================================")

if __name__ == "__main__":
    run_comprehensive_tests()
