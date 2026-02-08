import sys
import os
import random

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def inspect_db():
    print("=== 知识库健康度检查 (Step 1: Start) ===", flush=True)
    
    try:
        print("正在导入 VectorStoreHandler... (这可能需要几秒钟)", flush=True)
        from src.vector_store import VectorStoreHandler
        print("导入成功！", flush=True)
        
        # Initialize handler
        print("正在初始化 VectorStoreHandler...", flush=True)
        vs = VectorStoreHandler()
        print("初始化成功！", flush=True)
        
        # 1. Check Collection Count
        print("正在统计数据...", flush=True)
        count = vs.vectordb._collection.count()
        print(f"[1] 知识库总片段数 (Chunks): {count}")
        
        if count == 0:
            print("⚠️ 警告: 知识库为空！请检查是否已执行 ingest_data.py 或在界面上传文档。")
            return

        # 2. Random Sampling (Check Quality)
        print("\n[2] 随机抽样检查 (检查是否存在乱码/切分过碎):")
        all_ids = vs.vectordb._collection.get()['ids']
        sample_ids = random.sample(all_ids, min(3, count))
        samples = vs.vectordb._collection.get(ids=sample_ids)
        
        for i, (doc_id, content, meta) in enumerate(zip(samples['ids'], samples['documents'], samples['metadatas'])):
            print(f"--- 样本 {i+1} (ID: {doc_id}) ---")
            print(f"来源: {meta.get('source', 'Unknown')} (Page {meta.get('page', '?')})")
            print(f"内容预览 (前100字): {content[:100].replace(chr(10), ' ')}...") 
            print("---------------------------")

        # 3. Retrieval Test (Check Effectiveness)
        test_queries = ["声纳方程", "水下噪声", "多途效应"]
        print(f"\n[3] 检索能力测试 (测试词: {', '.join(test_queries)}):")
        
        for query in test_queries:
            print(f"\n🔍 搜索: '{query}'")
            results = vs.search(query, k=2)
            if not results:
                print("   ❌ 未找到相关文档")
            else:
                for j, doc in enumerate(results):
                    print(f"   ✅ 命中 {j+1}: {doc.page_content[:50].replace(chr(10), ' ')}... [来源: {doc.metadata.get('source')}]")

    except Exception as e:
        print(f"\n❌ 检查过程中发生错误: {e}")

if __name__ == "__main__":
    inspect_db()
