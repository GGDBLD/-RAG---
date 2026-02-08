import os
import sys

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def manual_test():
    print("=== 开始手动测试 VectorStore 模块 ===", flush=True)
    
    try:
        from src.vector_store import VectorStoreHandler
        import docx
        
        # 1. 初始化
        print("1. 正在初始化 VectorStoreHandler...", flush=True)
        vs = VectorStoreHandler()
        print("   初始化完成。", flush=True)
        
        # 2. 创建临时文件
        test_file = "manual_test_doc.docx"
        print(f"2. 创建测试文件: {test_file}", flush=True)
        doc = docx.Document()
        doc.add_paragraph("这是手动测试文档。核心关键词是：贝塔声纳阵列的高灵敏度接收。")
        doc.save(test_file)
        abs_path = os.path.abspath(test_file)
        
        # 3. 添加文档
        print("3. 调用 add_document...", flush=True)
        success, msg, chunks = vs.add_document(abs_path, "core")
        if success:
            print(f"   ✅ 添加成功! 片段数: {chunks}", flush=True)
        else:
            print(f"   ❌ 添加失败: {msg}", flush=True)
            return

        # 4. 搜索测试
        query = "贝塔声纳"
        print(f"4. 测试搜索: '{query}'", flush=True)
        results = vs.search(query, k=1)
        
        if results:
            print(f"   ✅ 搜索命中! 内容: {results[0].page_content}", flush=True)
            if "贝塔声纳" in results[0].page_content:
                print("   ✅ 内容匹配验证通过。", flush=True)
            else:
                print("   ❌ 内容匹配验证失败。", flush=True)
        else:
            print("   ❌ 搜索无结果。", flush=True)
            
        # 5. 清理
        print("5. 清理临时文件...", flush=True)
        if os.path.exists(test_file):
            os.remove(test_file)
        print("=== 测试结束 ===", flush=True)

    except Exception as e:
        print(f"\n❌ 测试过程中发生异常: {e}", flush=True)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    manual_test()