
import time
import sys
import os

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

print("🔄 Initializing System Components...")
try:
    from src.qa_chain import qa_chain
    print("✅ QA Chain loaded successfully.")
except Exception as e:
    print(f"❌ Failed to load QA Chain: {e}")
    sys.exit(1)

def test_question(question, history=None, expected_keywords=None, is_rule_based=False):
    print(f"\n🔹 Testing Question: '{question}'")
    if history:
        print(f"   Context: {len(history)} previous turns")
    
    start_time = time.time()
    try:
        # Mocking vector store search if needed? No, use real one.
        answer, sources = qa_chain.answer_question(question, chat_history=history)
        duration = time.time() - start_time
        
        print(f"   ⏱️ Duration: {duration:.2f}s")
        print(f"   📝 Answer Length: {len(answer)} chars")
        print(f"   📄 Sources: {len(sources)} docs")
        
        # Validation
        passed = True
        if not answer or "发生错误" in answer:
            print(f"   ❌ Error in answer: {answer}")
            passed = False
        
        if expected_keywords:
            missing = [k for k in expected_keywords if k not in answer]
            if missing:
                print(f"   ⚠️ Warning: Missing keywords {missing}")
        
        if is_rule_based and duration > 2.0:
             print("   ⚠️ Warning: Rule-based answer took too long")

        if passed:
            print("   ✅ Test Passed")
        
        return answer, passed, duration

    except Exception as e:
        print(f"   ❌ Exception: {e}")
        return None, False, 0

def run_tests():
    print("🚀 Starting Automated System Test...")
    
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
    print(f"📊 Test Summary")
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Total Duration: {total_duration:.2f}s")
    print("="*30)

if __name__ == "__main__":
    run_tests()
    input("\nPress Enter to exit...")
